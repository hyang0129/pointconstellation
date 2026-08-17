"""Run the pinned pcc_geo_cnn_v2 release on Experiment 019 source clouds."""

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

from pointconstellation.codecs import (
    ExternalCodecSpec,
    run_external_codec_batch,
    run_pc_error,
)
from pointconstellation.data import file_sha256
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _per_cloud_chamfer,
)


@dataclass(frozen=True)
class PccGeoCnnV2Manifest:
    """Pinned upstream and released-model protocol."""

    name: str
    upstream_url: str
    upstream_commit: str
    license: str
    paper: str
    paper_url: str
    artifact_record: str
    artifact_file: str
    artifact_bytes: int
    artifact_md5: str
    workspace_root: str
    upstream_subpath: str
    environment_python_subpath: str
    environment_manifest_subpath: str
    checkpoint_root_subpath: str
    model_mode: str
    model_config: str
    lambdas: tuple[str, ...]
    optimization_metric: str
    position_bits: int
    resolution: int
    octree_level: int
    timeout_seconds: float
    protocol_note: str
    checkout_diff_sha256: str | None = None
    training_manifest_subpath: str | None = None
    training_manifest_sha256: str | None = None
    training_arm: str | None = None

    def __post_init__(self) -> None:
        if self.name != "pcc_geo_cnn_v2":
            raise ValueError("this runner supports the pinned pcc_geo_cnn_v2 release")
        if len(self.upstream_commit) != 40:
            raise ValueError("upstream commit must be a full SHA")
        if self.license != "MIT":
            raise ValueError("unexpected upstream license declaration")
        if self.artifact_bytes < 1 or len(self.artifact_md5) != 32:
            raise ValueError("released artifact identity is invalid")
        if len(self.lambdas) < 5 or len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("at least five unique lambda points are required")
        if self.optimization_metric not in {"d1_mse", "d2_mse"}:
            raise ValueError("optimization_metric must be d1_mse or d2_mse")
        if not 2 <= self.position_bits <= 24:
            raise ValueError("position_bits must be between 2 and 24")
        if self.resolution != (1 << self.position_bits):
            raise ValueError("resolution must equal the declared coordinate grid size")
        if self.octree_level < 0 or self.timeout_seconds <= 0:
            raise ValueError("octree level and timeout are invalid")
        training_fields = (
            self.training_manifest_subpath,
            self.training_manifest_sha256,
            self.training_arm,
        )
        if any(field is not None for field in training_fields) and not all(
            field is not None for field in training_fields
        ):
            raise ValueError("retrained codec must declare its full training identity")
        if (
            self.training_manifest_sha256 is not None
            and len(self.training_manifest_sha256) != 64
        ):
            raise ValueError("training manifest must be pinned by SHA-256")

    @classmethod
    def from_json(cls, path: Path) -> PccGeoCnnV2Manifest:
        values = json.loads(path.read_text())
        values["lambdas"] = tuple(values["lambdas"])
        return cls(**values)


@dataclass(frozen=True)
class PublishedCodecBenchmarkConfig:
    """Configuration for an actual-stream published-codec evaluation."""

    codec_manifest: str
    stability_config: str
    pc_error_executable: str
    splits: tuple[str, ...] = ("validation", "ood")
    max_clouds_per_split: int | None = None
    workspace_root: str | None = None
    output_dir: str = "artifacts/local/experiment_020_pcc_geo_cnn_v2"

    def __post_init__(self) -> None:
        if not self.splits or set(self.splits) - {"validation", "ood"}:
            raise ValueError("splits must contain validation and/or ood")
        if self.max_clouds_per_split is not None and self.max_clouds_per_split < 1:
            raise ValueError("max_clouds_per_split must be positive")

    @classmethod
    def from_json(cls, path: Path) -> PublishedCodecBenchmarkConfig:
        values = json.loads(path.read_text())
        if "splits" in values:
            values["splits"] = tuple(values["splits"])
        return cls(**values)


def _directory_bytes(path: Path) -> int:
    if not path.is_dir():
        raise FileNotFoundError(f"external checkpoint directory is missing: {path}")
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    if not files:
        raise RuntimeError(f"external checkpoint directory is empty: {path}")
    return sum(candidate.stat().st_size for candidate in files)


def _trained_checkpoint_bytes(path: Path) -> int:
    state = path / "checkpoint"
    if not state.is_file():
        raise FileNotFoundError(f"TensorFlow checkpoint state is missing: {state}")
    first_line = state.read_text().splitlines()[0]
    if '"' not in first_line:
        raise RuntimeError(f"TensorFlow checkpoint state is invalid: {state}")
    prefix = first_line.split('"', 2)[1]
    files = [state, *sorted(path.glob(f"{prefix}.*"))]
    if len(files) < 3 or any(not candidate.is_file() for candidate in files):
        raise RuntimeError(f"TensorFlow deployment checkpoint is incomplete: {path}")
    return sum(candidate.stat().st_size for candidate in files)


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["lambda"],
        row["split"],
        row["family"],
        row["model_id"],
        row["sample_id"],
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("published-codec artifact contains duplicate rows")
    return rows


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _rate_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["lambda"], row["split"]), []).append(row)
    summaries = []
    for (rate_lambda, split), group in sorted(groups.items()):
        summaries.append(
            {
                "lambda": rate_lambda,
                "split": split,
                "clouds": len(group),
                "mean_stream_bytes": float(
                    np.mean([row["stream_bytes"] for row in group])
                ),
                "mean_actual_bpp": float(
                    np.mean([row["actual_stream_bpp"] for row in group])
                ),
                "mean_actual_bpov": float(
                    np.mean([row["actual_stream_bpov"] for row in group])
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
                "mean_d1_psnr_db": float(np.mean([row["d1_psnr_db"] for row in group])),
                "mean_d2_psnr_db": float(np.mean([row["d2_psnr_db"] for row in group])),
                "mean_encode_seconds": float(
                    np.mean([row["encode_seconds"] for row in group])
                ),
                "mean_official_metric_seconds": float(
                    np.mean([row["official_metric_seconds"] for row in group])
                ),
                "model_bytes": group[0]["model_bytes"],
            }
        )
    return summaries


def _codec_spec(
    manifest: PccGeoCnnV2Manifest,
    *,
    workspace: Path,
    rate_lambda: str,
) -> ExternalCodecSpec:
    upstream = workspace / manifest.upstream_subpath
    python = workspace / manifest.environment_python_subpath
    environment = workspace / manifest.environment_manifest_subpath
    checkpoint = (
        workspace / manifest.checkpoint_root_subpath / manifest.model_mode / rate_lambda
    )
    if not python.is_file() or not python.stat().st_mode & 0o111:
        raise FileNotFoundError(f"external environment Python is missing: {python}")
    model_bytes = (
        _trained_checkpoint_bytes(checkpoint)
        if manifest.training_manifest_subpath is not None
        else _directory_bytes(checkpoint)
    )
    if manifest.training_manifest_subpath is not None:
        training_manifest = workspace / manifest.training_manifest_subpath
        if not training_manifest.is_file() or file_sha256(training_manifest) != (
            manifest.training_manifest_sha256
        ):
            raise RuntimeError("external retraining manifest identity mismatch")
        training_run = checkpoint / "training_run.json"
        if not training_run.is_file():
            raise FileNotFoundError(
                f"external training record is missing: {training_run}"
            )
        training_record = json.loads(training_run.read_text())
        expected = {
            "arm": manifest.training_arm,
            "lambda": rate_lambda,
            "upstream_commit": manifest.upstream_commit,
            "dataset_manifest_sha256": manifest.training_manifest_sha256,
        }
        if any(training_record.get(key) != value for key, value in expected.items()):
            raise RuntimeError("external checkpoint training identity mismatch")
    compress_command = (
        str(python),
        "{upstream_dir}/src/compress_octree.py",
        "--input_files",
        "{inputs}",
        "--output_files",
        "{streams}",
        "--checkpoint_dir",
        "{checkpoint_dir}",
        "--opt_metrics",
        manifest.optimization_metric,
        "--resolution",
        str(manifest.resolution),
        "--model_config",
        manifest.model_config,
        "--octree_level",
        str(manifest.octree_level),
    )
    decompress_command = (
        str(python),
        "{upstream_dir}/src/decompress_octree.py",
        "--input_files",
        "{streams}",
        "--output_files",
        "{reconstructions}",
        "--checkpoint_dir",
        "{checkpoint_dir}",
        "--model_config",
        manifest.model_config,
    )
    return ExternalCodecSpec(
        name=f"{manifest.name}_{manifest.model_mode}_{rate_lambda}",
        upstream_url=manifest.upstream_url,
        upstream_commit=manifest.upstream_commit,
        upstream_dir=str(upstream),
        checkpoint_dir=str(checkpoint),
        compress_command=compress_command,
        decompress_command=decompress_command,
        position_bits=manifest.position_bits,
        timeout_seconds=manifest.timeout_seconds,
        model_bytes=model_bytes,
        environment_manifest=str(environment),
        environment_variables=(
            (
                "LD_LIBRARY_PATH",
                str(workspace / "env" / "lib")
                + (
                    f":{os.environ['LD_LIBRARY_PATH']}"
                    if os.environ.get("LD_LIBRARY_PATH")
                    else ""
                ),
            ),
            (
                "CUDA_VISIBLE_DEVICES",
                os.environ.get("POINTCONSTELLATION_CODEC_GPU", "0"),
            ),
            ("CUDA_CACHE_MAXSIZE", str(2 * 1024**3)),
            ("TF_CUDNN_USE_AUTOTUNE", "0"),
        ),
        checkout_diff_sha256=manifest.checkout_diff_sha256,
        allow_empty_reconstruction=manifest.training_manifest_subpath is not None,
    )


def run_published_codec_benchmark(
    config: PublishedCodecBenchmarkConfig,
) -> dict[str, Any]:
    """Run and resume all released-model rate points on fixed source clouds."""

    manifest_path = Path(config.codec_manifest)
    manifest = PccGeoCnnV2Manifest.from_json(manifest_path)
    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    if manifest.position_bits > stability.coordinate_bits:
        raise ValueError("external codec precision exceeds the common metric grid")
    workspace = Path(config.workspace_root or manifest.workspace_root)
    executable = Path(config.pc_error_executable)
    if not executable.is_file() or not executable.stat().st_mode & 0o111:
        raise FileNotFoundError(f"pc_error is missing or not executable: {executable}")
    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = output_dir / "metric_scratch"
    scratch.mkdir(exist_ok=True)
    run_manifest = {
        "experiment": "020_pcc_geo_cnn_v2",
        "config": json.loads(json.dumps(asdict(config))),
        "codec_manifest": json.loads(manifest_path.read_text()),
        "codec_manifest_sha256": file_sha256(manifest_path),
        "stability_config_sha256": file_sha256(stability_path),
        "pc_error_sha256": file_sha256(executable),
        "data_protocol": data_protocol,
    }
    run_manifest_path = output_dir / "run_manifest.json"
    if run_manifest_path.exists():
        if json.loads(run_manifest_path.read_text()) != run_manifest:
            raise RuntimeError("existing external benchmark manifest does not match")
    else:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")

    rows_path = output_dir / "per_cloud.jsonl"
    rows = _load_rows(rows_path)
    resumed_rows = len(rows)
    completed = {_row_key(row) for row in rows}
    started = time.perf_counter()
    model_records = []
    for rate_lambda in manifest.lambdas:
        spec = _codec_spec(manifest, workspace=workspace, rate_lambda=rate_lambda)
        model_records.append(
            {
                "lambda": rate_lambda,
                "name": spec.name,
                "model_bytes": spec.model_bytes,
                "checkpoint_dir": spec.checkpoint_dir,
                "upstream_commit": spec.upstream_commit,
            }
        )
        pending: list[dict[str, Any]] = []
        for split in config.splits:
            dataset = datasets[split]
            count = (
                len(dataset)
                if config.max_clouds_per_split is None
                else min(len(dataset), config.max_clouds_per_split)
            )
            for sample_index in range(count):
                sample = dataset[sample_index]
                key = (
                    rate_lambda,
                    split,
                    str(sample["family"]),
                    str(sample["model_id"]),
                    int(sample["sample_id"]),
                )
                if key in completed:
                    continue
                cloud_dir = (
                    output_dir
                    / "streams"
                    / f"lambda_{rate_lambda}"
                    / split
                    / f"{sample['family']}_{sample['model_id']}"
                )
                pending.append(
                    {
                        "key": key,
                        "split": split,
                        "sample": sample,
                        "cloud_dir": cloud_dir,
                        "source": sample["source_points"].numpy(),
                        "normals": sample["source_normals"].numpy(),
                    }
                )
        if not pending:
            continue
        batch_results = run_external_codec_batch(
            spec,
            tuple(item["source"] for item in pending),
            work_dirs=tuple(item["cloud_dir"] for item in pending),
        )
        batch_log_dir = output_dir / "streams" / f"lambda_{rate_lambda}"
        batch_log_dir.mkdir(parents=True, exist_ok=True)
        (batch_log_dir / "compress.log").write_text(batch_results[0].compress_output)
        (batch_log_dir / "decompress.log").write_text(
            batch_results[0].decompress_output
        )
        for item, codec_result in zip(pending, batch_results, strict=True):
            sample = item["sample"]
            split = item["split"]
            cloud_dir = item["cloud_dir"]
            source = item["source"]
            normals = item["normals"]
            codec_levels = (1 << manifest.position_bits) - 1
            codec_input = np.rint((source + 1.0) * 0.5 * codec_levels)
            codec_input_unique_voxels = len(np.unique(codec_input, axis=0))
            reconstruction = torch.from_numpy(codec_result.reconstruction).unsqueeze(0)
            chamfer_mse = float(
                _per_cloud_chamfer(
                    reconstruction,
                    sample["source_points"].unsqueeze(0),
                    chunk_size=stability.distance_chunk_size,
                )[0].item()
            )
            with tempfile.TemporaryDirectory(
                prefix=f"{split}-{rate_lambda}-", dir=scratch
            ) as temporary:
                official = run_pc_error(
                    executable,
                    source,
                    codec_result.reconstruction,
                    normals,
                    work_dir=Path(temporary),
                    position_bits=stability.coordinate_bits,
                )
            row = {
                "codec": manifest.name,
                "model_mode": manifest.model_mode,
                "lambda": rate_lambda,
                "optimization_metric": manifest.optimization_metric,
                "split": split,
                "family": str(sample["family"]),
                "model_id": str(sample["model_id"]),
                "sample_id": int(sample["sample_id"]),
                "source_points": len(source),
                "codec_input_unique_voxels": codec_input_unique_voxels,
                "decoded_points": len(codec_result.reconstruction),
                "stream_bytes": codec_result.stream_bytes,
                "actual_stream_bpp": (8.0 * codec_result.stream_bytes / len(source)),
                "actual_stream_bpov": (
                    8.0 * codec_result.stream_bytes / codec_input_unique_voxels
                ),
                "codec_input_position_bits": manifest.position_bits,
                "metric_position_bits": stability.coordinate_bits,
                "stream_sha256": codec_result.stream_sha256,
                "reconstruction_sha256": codec_result.reconstruction_sha256,
                "checkout_diff_sha256": codec_result.checkout_diff_sha256,
                "encode_seconds": codec_result.encode_seconds,
                "decode_seconds": codec_result.decode_seconds,
                "external_batch_clouds": len(pending),
                "official_metric_seconds": official.elapsed_seconds,
                "model_bytes": spec.model_bytes,
                "chamfer_mse": chamfer_mse,
                **official.metrics,
            }
            (cloud_dir / "row.json").write_text(json.dumps(row, indent=2) + "\n")
            _append_row(rows_path, row)
            rows.append(row)
            completed.add(item["key"])

    result = {
        "experiment": "020_pcc_geo_cnn_v2",
        "config": json.loads(json.dumps(asdict(config))),
        "codec_manifest": json.loads(manifest_path.read_text()),
        "resumed_rows": resumed_rows,
        "per_cloud_rows": len(rows),
        "model_records": model_records,
        "rate_summaries": _rate_summaries(rows),
        "contract_checks": {
            "actual_streams_nonempty": bool(
                rows and all(row["stream_bytes"] > 0 for row in rows)
            ),
            "stream_hashes_present": bool(
                rows and all(len(row["stream_sha256"]) == 64 for row in rows)
            ),
            "upstream_commit_pinned": True,
            "source_partition_unchanged": True,
            "copied_paper_values_absent": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "per_cloud_path": str(rows_path),
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError("published-codec benchmark contract failed")
    metrics_path = output_dir / "published_codec_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "rows": result["per_cloud_rows"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(metrics_path),
            }
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--max-clouds-per-split", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = PublishedCodecBenchmarkConfig.from_json(args.config)
    if args.workspace_root is not None:
        config = replace(config, workspace_root=str(args.workspace_root))
    if args.max_clouds_per_split is not None:
        config = replace(config, max_clouds_per_split=args.max_clouds_per_split)
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    run_published_codec_benchmark(config)


if __name__ == "__main__":
    main()
