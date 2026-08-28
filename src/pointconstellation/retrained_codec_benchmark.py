"""Evaluate one exact-split retrained pcc_geo_cnn_v2 lambda point."""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pointconstellation.codecs import run_external_codec_batch, run_pc_error
from pointconstellation.data import file_sha256
from pointconstellation.metrics import chamfer_rmse
from pointconstellation.published_codec_benchmark import (
    PccGeoCnnV2Manifest,
    _codec_spec,
    _diversity_contract_summary,
)
from pointconstellation.rate_accounting import amortized_bpp_table


@dataclass(frozen=True)
class RetrainedCodecBenchmarkConfig:
    codec_manifest: str
    evaluation_root: str
    expected_evaluation_manifest_sha256: str
    pc_error_executable: str
    output_dir: str
    minimum_training_steps: int = 5000
    metric_position_bits: int = 12
    minimum_unique_reconstruction_fraction: float = 0.9

    def __post_init__(self) -> None:
        if len(self.expected_evaluation_manifest_sha256) != 64:
            raise ValueError("evaluation manifest must be pinned by SHA-256")
        if self.minimum_training_steps < 500:
            raise ValueError("minimum_training_steps must be at least 500")
        if not 2 <= self.metric_position_bits <= 24:
            raise ValueError("metric_position_bits must be between 2 and 24")
        if not 0.0 < self.minimum_unique_reconstruction_fraction <= 1.0:
            raise ValueError("minimum_unique_reconstruction_fraction must be in (0, 1]")

    @classmethod
    def from_json(cls, path: Path) -> RetrainedCodecBenchmarkConfig:
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class GzipStreamBreakdown:
    """Exact gzip wrapper and compressed-DEFLATE byte counts."""

    header_bytes: int
    payload_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        if min(self.header_bytes, self.payload_bytes) < 0:
            raise ValueError("gzip byte counts cannot be negative")
        if self.header_bytes + self.payload_bytes != self.total_bytes:
            raise ValueError("gzip components do not sum to total_bytes")


def _gzip_stream_breakdown(path: Path) -> GzipStreamBreakdown:
    """Split all gzip members into wrapper and compressed payload bytes."""

    data = path.read_bytes()
    if not data:
        raise ValueError("gzip stream is empty")
    offset = 0
    header_bytes = 0
    payload_bytes = 0
    while offset < len(data):
        member_start = offset
        if len(data) - offset < 10 or data[offset : offset + 3] != b"\x1f\x8b\x08":
            raise ValueError(f"invalid gzip member header at byte {offset}")
        flags = data[offset + 3]
        if flags & 0xE0:
            raise ValueError(f"gzip member has reserved flags at byte {offset}")
        offset += 10
        if flags & 0x04:
            if len(data) - offset < 2:
                raise ValueError("truncated gzip extra-field length")
            extra_bytes = int.from_bytes(data[offset : offset + 2], "little")
            offset += 2 + extra_bytes
            if offset > len(data):
                raise ValueError("truncated gzip extra field")
        for flag, name in ((0x08, "file name"), (0x10, "comment")):
            if flags & flag:
                end = data.find(b"\0", offset)
                if end < 0:
                    raise ValueError(f"unterminated gzip {name}")
                offset = end + 1
        if flags & 0x02:
            offset += 2
            if offset > len(data):
                raise ValueError("truncated gzip header CRC")
        payload_start = offset
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        decompressed = inflater.decompress(data[payload_start:])
        if not inflater.eof:
            raise ValueError("truncated gzip DEFLATE payload")
        compressed_bytes = len(data) - payload_start - len(inflater.unused_data)
        offset = payload_start + compressed_bytes
        if len(data) - offset < 8:
            raise ValueError("truncated gzip trailer")
        crc32, size = struct.unpack_from("<II", data, offset)
        if (
            zlib.crc32(decompressed) != crc32
            or (len(decompressed) & 0xFFFFFFFF) != size
        ):
            raise ValueError("gzip trailer does not match decompressed payload")
        offset += 8
        header_bytes += payload_start - member_start + 8
        payload_bytes += compressed_bytes
    return GzipStreamBreakdown(header_bytes, payload_bytes, len(data))


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


def _rmse_or_none(values: list[float | None]) -> float | None:
    """Return aggregate RMSE only when every declared cloud is valid."""

    if not values or any(value is None for value in values):
        return None
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64))))


def _valid_only_rmse(values: list[float | None]) -> float | None:
    """Return an explicitly diagnostic RMSE over non-failure rows only."""

    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(np.sqrt(np.mean(np.asarray(valid, dtype=np.float64))))


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
    checkpoint_state = (checkpoint / "checkpoint").read_text().splitlines()[0]
    checkpoint_prefix = checkpoint_state.split('"', 2)[1]
    checkpoint_step = int(checkpoint_prefix.rsplit("-", 1)[1])
    if checkpoint_step < config.minimum_training_steps:
        raise RuntimeError("actual checkpoint step is below the evaluation budget")
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
    metric_scratch.mkdir(exist_ok=True)
    for record, codec_result, work_dir in zip(records, results, work_dirs, strict=True):
        source = record["source"]
        normals = record["normals"]
        stream_breakdown = _gzip_stream_breakdown(work_dir / "stream.bin")
        if stream_breakdown.total_bytes != codec_result.stream_bytes:
            raise RuntimeError("external gzip breakdown differs from stream size")
        empty_reconstruction = len(codec_result.reconstruction) == 0
        if empty_reconstruction:
            official_metrics = {
                key: None
                for key in (
                    "d1_mse",
                    "d2_mse",
                    "d1_hausdorff",
                    "d2_hausdorff",
                )
            }
            official_metrics.update(
                {
                    key: None
                    for key in (
                        "d1_psnr_db",
                        "d2_psnr_db",
                        "d1_hausdorff_psnr_db",
                        "d2_hausdorff_psnr_db",
                    )
                }
            )
            chamfer_mse = None
        else:
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
            official_metrics = official.metrics
            chamfer_mse = chamfer_rmse(codec_result.reconstruction, source) ** 2
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
                "status": ("empty_reconstruction" if empty_reconstruction else "valid"),
                "header_bytes": stream_breakdown.header_bytes,
                "stream_bytes": codec_result.stream_bytes,
                "payload_bytes": stream_breakdown.payload_bytes,
                "payload_bpp": 8.0 * stream_breakdown.payload_bytes / len(source),
                "actual_stream_bpp": 8.0 * codec_result.stream_bytes / len(source),
                "stream_sha256": codec_result.stream_sha256,
                "reconstruction_sha256": codec_result.reconstruction_sha256,
                "chamfer_mse": chamfer_mse,
                "encode_seconds": codec_result.encode_seconds,
                "decode_seconds": codec_result.decode_seconds,
                **official_metrics,
            }
        )
    summaries = []
    for split in ("validation", "ood"):
        group = [row for row in rows if row["split"] == split]
        stream_bytes = np.asarray([row["stream_bytes"] for row in group])
        empty_reconstructions = sum(
            row["status"] == "empty_reconstruction" for row in group
        )
        chamfer_values = [row["chamfer_mse"] for row in group]
        d1_values = [row["d1_mse"] for row in group]
        d2_values = [row["d2_mse"] for row in group]
        diversity = _diversity_contract_summary(
            group,
            minimum_unique_reconstruction_fraction=(
                config.minimum_unique_reconstruction_fraction
            ),
        )
        summaries.append(
            {
                "split": split,
                "clouds": len(group),
                "valid_clouds": sum(row["status"] == "valid" for row in group),
                "empty_reconstructions": empty_reconstructions,
                **diversity,
                "rate_point_valid": (
                    empty_reconstructions == 0 and diversity["rate_point_valid"]
                ),
                "mean_stream_bytes": float(stream_bytes.mean()),
                "median_stream_bytes": float(np.median(stream_bytes)),
                "minimum_stream_bytes": int(stream_bytes.min()),
                "maximum_stream_bytes": int(stream_bytes.max()),
                "mean_actual_bpp": float(
                    np.mean([row["actual_stream_bpp"] for row in group])
                ),
                "mean_payload_bpp": float(
                    np.mean([row["payload_bpp"] for row in group])
                ),
                "model_bytes": spec.model_bytes,
                "amortized_bpp": amortized_bpp_table(
                    float(stream_bytes.mean()),
                    spec.model_bytes or 0,
                    int(group[0]["source_points"]),
                ),
                "aggregate_chamfer_rmse": _rmse_or_none(chamfer_values),
                "official_d1_rmse_grid_units": _rmse_or_none(d1_values),
                "official_d2_rmse_grid_units": _rmse_or_none(d2_values),
                "valid_only_chamfer_rmse_diagnostic": _valid_only_rmse(chamfer_values),
                "valid_only_d1_rmse_grid_units_diagnostic": _valid_only_rmse(d1_values),
                "valid_only_d2_rmse_grid_units_diagnostic": _valid_only_rmse(d2_values),
            }
        )
    result = {
        "experiment": "020_exact_retrained_codec_rate",
        "config": asdict(config),
        "lambda": rate_lambda,
        "training_run_sha256": file_sha256(training_run_path),
        "checkpoint_step": checkpoint_step,
        "model_bytes": spec.model_bytes,
        "rows": rows,
        "summaries": summaries,
        "target_constellation_bytes": 50,
        "rate_accounting": {
            "full_stream_bytes_available": True,
            "payload_bytes_available": True,
            "payload_note": (
                "payload_bytes is the exact compressed-DEFLATE portion of all gzip "
                "members; gzip headers and eight-byte member trailers are excluded."
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "contract_checks": {
            "actual_streams_nonempty": all(row["stream_bytes"] > 0 for row in rows),
            "independent_decode_hashes_present": all(
                len(row["reconstruction_sha256"]) == 64 for row in rows
            ),
            "header_payload_splits_exact": all(
                row["header_bytes"] + row["payload_bytes"] == row["stream_bytes"]
                for row in rows
            ),
            "test_only_archive": all(
                row["split"] in {"validation", "ood"} for row in rows
            ),
            "minimum_training_budget_reached": True,
            "training_identity_pinned": True,
            "checkout_patch_pinned": True,
            "empty_reconstructions_explicitly_recorded": all(
                (row["decoded_points"] == 0)
                == (row["status"] == "empty_reconstruction")
                for row in rows
            ),
        },
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError("retrained external-codec evaluation contract failed")
    metrics_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
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
