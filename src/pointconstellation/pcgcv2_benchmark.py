"""Low-rate PCGCv2 black-box evaluation helpers for Experiment 027."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pointconstellation.codecs import (
    Pcgcv2HarnessConfig,
    pcgcv2_diversity_summary,
    point_set_sha256,
    run_external_codec,
    run_pc_error,
)
from pointconstellation.data import file_sha256


def evaluate_pcgcv2_rate_point(
    harness: Pcgcv2HarnessConfig,
    rate_point: str,
    position_bits: int,
    source: ArrayLike,
    normals: ArrayLike,
    *,
    pc_error_executable: Path,
    work_dir: Path,
    split: str,
    family: str,
    model_id: str,
    sample_id: int,
    metric_position_bits: int = 12,
    metric_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Run one complete stream and return a registry-ready per-cloud row."""

    source_array = np.asarray(source, dtype=np.float32)
    normal_array = np.asarray(normals, dtype=np.float32)
    if source_array.ndim != 2 or source_array.shape[1] != 3 or not len(source_array):
        raise ValueError("PCGCv2 source must have shape (N, 3) with N > 0")
    if source_array.shape != normal_array.shape:
        raise ValueError("PCGCv2 source points and normals must have matching shapes")
    if not 2 <= metric_position_bits <= 24:
        raise ValueError("metric_position_bits must be between 2 and 24")
    spec = harness.external_spec(rate_point, position_bits)
    result = run_external_codec(spec, source_array, work_dir=work_dir / "codec")
    with tempfile.TemporaryDirectory(prefix="metric-", dir=work_dir) as temporary:
        official = run_pc_error(
            pc_error_executable,
            source_array,
            result.reconstruction,
            normal_array,
            work_dir=Path(temporary),
            position_bits=metric_position_bits,
            timeout_seconds=metric_timeout_seconds,
        )
    if spec.model_bytes is None:
        raise RuntimeError("PCGCv2 deployment checkpoint size was not counted")
    return {
        "experiment": "027_pcgcv2",
        "dataset_split": split,
        "split": split,
        "family": family,
        "model_id": model_id,
        "sample_id": sample_id,
        "codec_family": "pcgcv2",
        "method": spec.name,
        "rate_point": rate_point,
        "source_points": len(source_array),
        "decoded_points": len(result.reconstruction),
        "codec_input_position_bits": position_bits,
        "metric_position_bits": metric_position_bits,
        "payload_bytes": result.stream_bytes,
        "stream_bytes": result.stream_bytes,
        "actual_stream_bpp": 8.0 * result.stream_bytes / len(source_array),
        "model_bytes": spec.model_bytes,
        "model_sha256": file_sha256(Path(spec.checkpoint_dir)),
        "stream_sha256": result.stream_sha256,
        "reconstruction_sha256": result.reconstruction_sha256,
        "reconstruction_geometry_sha256": point_set_sha256(result.reconstruction),
        "upstream_commit": result.upstream_commit,
        "checkout_diff_sha256": result.checkout_diff_sha256,
        "encode_seconds": result.encode_seconds,
        "decode_seconds": result.decode_seconds,
        "official_metric_seconds": official.elapsed_seconds,
        **official.metrics,
    }


def summarize_pcgcv2_rows(
    rows: list[dict[str, Any]],
    *,
    minimum_unique_reconstruction_fraction: float = 0.9,
) -> list[dict[str, Any]]:
    """Summarize actual rates and apply the diversity validity contract."""

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["rate_point"]),
            int(row["codec_input_position_bits"]),
            str(row["split"]),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for (rate_point, position_bits, split), group in sorted(groups.items()):
        diversity = pcgcv2_diversity_summary(
            group,
            minimum_unique_reconstruction_fraction=(
                minimum_unique_reconstruction_fraction
            ),
        )
        summaries.append(
            {
                "rate_point": rate_point,
                "codec_input_position_bits": position_bits,
                "split": split,
                **diversity,
                "mean_payload_bytes": float(
                    np.mean([row["payload_bytes"] for row in group])
                ),
                "mean_actual_stream_bpp": float(
                    np.mean([row["actual_stream_bpp"] for row in group])
                ),
                "model_bytes": group[0]["model_bytes"],
                "official_d1_rmse_grid_units": math.sqrt(
                    float(np.mean([row["d1_mse"] for row in group]))
                ),
                "official_d2_rmse_grid_units": math.sqrt(
                    float(np.mean([row["d2_mse"] for row in group]))
                ),
            }
        )
    return summaries
