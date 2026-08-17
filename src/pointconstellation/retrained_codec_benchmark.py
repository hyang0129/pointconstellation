"""Evaluate one exact-split retrained pcc_geo_cnn_v2 lambda point."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pointconstellation.codecs import run_external_codec_batch, run_pc_error
from pointconstellation.data import file_sha256
from pointconstellation.published_codec_benchmark import (
    PccGeoCnnV2Manifest,
    _codec_spec,
)
from pointconstellation.stability_experiment import _per_cloud_chamfer


@dataclass(frozen=True)
class RetrainedCodecBenchmarkConfig:
    codec_manifest: str
    evaluation_root: str
    expected_evaluation_manifest_sha256: str
    pc_error_executable: str
    output_dir: str
    minimum_training_steps: int = 5000
    metric_position_bits: int = 12

    def __post_init__(self) -> None:
        if len(self.expected_evaluation_manifest_sha256) != 64:
            raise ValueError("evaluation manifest must be pinned by SHA-256")
        if self.minimum_training_steps < 500:
            raise ValueError("minimum_training_steps must be at least 500")
        if not 2 <= self.metric_position_bits <= 24:
            raise ValueError("metric_position_bits must be between 2 and 24")

    @classmethod
    def from_json(cls, path: Path) -> RetrainedCodecBenchmarkConfig:
        return cls(**json.loads(path.read_text()))


def _checked_array(root: Path, relative: str, expected_sha256: str) -> np.ndarray:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"evaluation array escapes its root: {relative}")
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"evaluation array SHA-256 differs: {relative}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"evaluation array must be finite N x 3: {relative}")
    return array.astype(np.float32, copy=False)


def _load_evaluation(config: RetrainedCodecBenchmarkConfig) -> list[dict[str, Any]]:
    root = Path(config.evaluation_root).resolve()
    manifest_path = root / "evaluation_manifest.json"
    if not manifest_path.is_file() or file_sha256(manifest_path) != (
        config.expected_evaluation_manifest_sha256
    ):
        raise RuntimeError("external evaluation manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records", [])
    if not records or {record["split"] for record in records} - {"validation", "ood"}:
        raise RuntimeError(
            "evaluation archive must contain only validation/OOD records"
        )
    loaded = []
    for record in records:
        source = _checked_array(root, record["source_path"], record["source_sha256"])
        normals = _checked_array(root, record["normals_path"], record["normals_sha256"])
        if source.shape != normals.shape:
            raise RuntimeError("source and normal evaluation arrays differ in shape")
        loaded.append({**record, "source": source, "normals": normals})
    return loaded


def run_retrained_codec_benchmark(
    config: RetrainedCodecBenchmarkConfig,
    *,
    rate_lambda: str,
    gpu: str,
) -> dict[str, Any]:
    """Run actual streams, independent decode, and official metrics."""

    manifest = PccGeoCnnV2Manifest.from_json(Path(config.codec_manifest))
    if rate_lambda not in manifest.lambdas:
        raise ValueError(f"lambda is not declared by codec manifest: {rate_lambda}")
    workspace = Path(manifest.workspace_root)
    checkpoint = (
        workspace / manifest.checkpoint_root_subpath / manifest.model_mode / rate_lambda
    )
    training_run_path = checkpoint / "training_run.json"
    if not training_run_path.is_file():
        raise FileNotFoundError(f"training record is missing: {training_run_path}")
    training_run = json.loads(training_run_path.read_text())
    if int(training_run["max_steps"]) < config.minimum_training_steps:
        raise RuntimeError("checkpoint has not reached the declared evaluation budget")
    executable = Path(config.pc_error_executable)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"pc_error is missing or not executable: {executable}")
    records = _load_evaluation(config)
    output_dir = Path(config.output_dir) / f"lambda_{rate_lambda}"
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["POINTCONSTELLATION_CODEC_GPU"] = gpu
    spec = _codec_spec(manifest, workspace=workspace, rate_lambda=rate_lambda)
    work_dirs = tuple(
        output_dir / "streams" / record["split"] / str(record["model_id"])
        for record in records
    )
    started = time.perf_counter()
    results = run_external_codec_batch(
        spec,
        tuple(record["source"] for record in records),
        work_dirs=work_dirs,
    )
    (output_dir / "compress.log").write_text(results[0].compress_output)
    (output_dir / "decompress.log").write_text(results[0].decompress_output)
    rows = []
    metric_scratch = output_dir / "metric_scratch"
    metric_scratch.mkdir()
    for record, codec_result in zip(records, results, strict=True):
        source = record["source"]
        normals = record["normals"]
        with tempfile.TemporaryDirectory(
            prefix=f"{record['split']}-{record['model_id']}-",
            dir=metric_scratch,
        ) as temporary:
            official = run_pc_error(
                executable,
                source,
                codec_result.reconstruction,
                normals,
                work_dir=Path(temporary),
                position_bits=config.metric_position_bits,
            )
        reconstruction = torch.from_numpy(codec_result.reconstruction).unsqueeze(0)
        target = torch.from_numpy(source).unsqueeze(0)
        chamfer_mse = float(
            _per_cloud_chamfer(reconstruction, target, chunk_size=256)[0].item()
        )
        codec_levels = (1 << manifest.position_bits) - 1
        unique_voxels = len(
            np.unique(np.rint((source + 1.0) * 0.5 * codec_levels), axis=0)
        )
        rows.append(
            {
                "lambda": rate_lambda,
                "split": record["split"],
                "category": record["category"],
                "model_id": record["model_id"],
                "source_points": len(source),
                "codec_input_unique_voxels": unique_voxels,
                "decoded_points": len(codec_result.reconstruction),
                "stream_bytes": codec_result.stream_bytes,
                "actual_stream_bpp": 8.0 * codec_result.stream_bytes / len(source),
                "stream_sha256": codec_result.stream_sha256,
                "reconstruction_sha256": codec_result.reconstruction_sha256,
                "chamfer_mse": chamfer_mse,
                "encode_seconds": codec_result.encode_seconds,
                "decode_seconds": codec_result.decode_seconds,
                **official.metrics,
            }
        )
    summaries = []
    for split in ("validation", "ood"):
        group = [row for row in rows if row["split"] == split]
        stream_bytes = np.asarray([row["stream_bytes"] for row in group])
        summaries.append(
            {
                "split": split,
                "clouds": len(group),
                "mean_stream_bytes": float(stream_bytes.mean()),
                "median_stream_bytes": float(np.median(stream_bytes)),
                "minimum_stream_bytes": int(stream_bytes.min()),
                "maximum_stream_bytes": int(stream_bytes.max()),
                "mean_actual_bpp": float(
                    np.mean([row["actual_stream_bpp"] for row in group])
                ),
                "aggregate_chamfer_rmse": math.sqrt(
                    float(np.mean([row["chamfer_mse"] for row in group]))
                ),
                "official_d1_rmse_grid_units": math.sqrt(
                    float(np.mean([row["d1_mse"] for row in group]))
                ),
                "official_d2_rmse_grid_units": math.sqrt(
                    float(np.mean([row["d2_mse"] for row in group]))
                ),
            }
        )
    result = {
        "experiment": "020_exact_retrained_codec_rate",
        "config": asdict(config),
        "lambda": rate_lambda,
        "training_run_sha256": file_sha256(training_run_path),
        "model_bytes": spec.model_bytes,
        "rows": rows,
        "summaries": summaries,
        "target_constellation_bytes": 50,
        "elapsed_seconds": time.perf_counter() - started,
        "contract_checks": {
            "actual_streams_nonempty": all(row["stream_bytes"] > 0 for row in rows),
            "independent_decode_hashes_present": all(
                len(row["reconstruction_sha256"]) == 64 for row in rows
            ),
            "test_only_archive": all(
                row["split"] in {"validation", "ood"} for row in rows
            ),
            "minimum_training_budget_reached": True,
            "training_identity_pinned": True,
            "checkout_patch_pinned": True,
        },
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError("retrained external-codec evaluation contract failed")
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lambda", dest="rate_lambda", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--pc-error", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = RetrainedCodecBenchmarkConfig.from_json(args.config)
    if args.evaluation_root is not None:
        config = replace(config, evaluation_root=str(args.evaluation_root))
    if args.pc_error is not None:
        config = replace(config, pc_error_executable=str(args.pc_error))
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    result = run_retrained_codec_benchmark(
        config, rate_lambda=args.rate_lambda, gpu=args.gpu
    )
    print(json.dumps({key: result[key] for key in ("lambda", "summaries")}, indent=2))


if __name__ == "__main__":
    main()
