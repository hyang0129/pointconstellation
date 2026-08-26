#!/usr/bin/env python3
"""Re-summarize existing standardized streams with header-normalized rates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.bitstream import (  # noqa: E402
    HEADER,
    expected_payload_bytes,
    expected_stream_bytes,
)
from pointconstellation.codecs.gpcc import parse_gpcc_stream  # noqa: E402
from pointconstellation.data import file_sha256  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark rows are missing: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean(group: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([row[field] for row in group]))


def _gpcc_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["split"], row["rate_point"]), []).append(row)
    result = []
    for (split, rate_point), group in sorted(groups.items()):
        fields = [
            "sps_bytes",
            "gps_bytes",
            "slice_header_bytes",
            "header_bytes",
            "payload_bytes",
            "stream_bytes",
            "payload_bpp",
            "actual_stream_bpp",
            "chamfer_rmse",
        ]
        if "amortized_stream_bytes" in group[0]:
            fields.extend(("amortized_stream_bytes", "amortized_stream_bpp"))
        summary = {
            "split": split,
            "rate_point": rate_point,
            "clouds": len(group),
            **{f"mean_{field}": _mean(group, field) for field in fields},
        }
        if "amortize_parameter_sets_over" in group[0]:
            summary["amortize_parameter_sets_over"] = group[0][
                "amortize_parameter_sets_over"
            ]
            summary["amortized_stream_note"] = group[0]["amortized_stream_note"]
        result.append(summary)
    return result


def _constellation_summary(
    rows: list[dict[str, Any]], *, input_points: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["split"], row["method"], row["constellation_size"])
        groups.setdefault(key, []).append(row)
    result = []
    for (split, method, size), group in sorted(groups.items()):
        bits = int(group[0]["coordinate_bits"])
        payload_bytes = expected_payload_bytes(size, bits)
        stream_bytes = expected_stream_bytes(size, bits)
        if any(row["stream_bytes"] != stream_bytes for row in group):
            raise RuntimeError("constellation row differs from exact stream size")
        result.append(
            {
                "split": split,
                "method": method,
                "constellation_size": size,
                "clouds": len(group),
                "header_bytes": HEADER.size,
                "payload_bytes": payload_bytes,
                "stream_bytes": stream_bytes,
                "payload_bpp": 8.0 * payload_bytes / input_points,
                "actual_stream_bpp": 8.0 * stream_bytes / input_points,
                "mean_chamfer_rmse": _mean(group, "chamfer_rmse"),
            }
        )
    return result


def resummarize(
    experiment_dir: Path, *, amortize_parameter_sets_over: int | None = None
) -> dict[str, Any]:
    """Return exact-rate summaries without modifying the source artifacts."""

    metrics_path = experiment_dir / "benchmark_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"benchmark metrics are missing: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    input_points = int(metrics["protocol"]["input_points"])
    gpcc_path = experiment_dir / "gpcc_per_cloud.jsonl"
    constellation_path = experiment_dir / "per_cloud.jsonl"
    gpcc_rows = _jsonl(gpcc_path)
    enriched_gpcc = []
    for row in gpcc_rows:
        stream_path = (
            experiment_dir
            / "gpcc_work"
            / row["split"]
            / f"sample_{int(row['sample_id']):05d}"
            / row["rate_point"]
            / "stream.bin"
        )
        breakdown = parse_gpcc_stream(stream_path)
        if breakdown.total_bytes != row["stream_bytes"]:
            raise RuntimeError(f"recorded G-PCC size differs from {stream_path}")
        enriched = {
            **row,
            **asdict(breakdown),
            "header_bytes": breakdown.header_bytes,
            "stream_bytes": breakdown.total_bytes,
            "payload_bpp": 8.0 * breakdown.payload_bytes / input_points,
        }
        if amortize_parameter_sets_over is not None:
            amortized_bytes = breakdown.amortized_stream_bytes(
                amortize_parameter_sets_over
            )
            enriched.update(
                {
                    "amortize_parameter_sets_over": amortize_parameter_sets_over,
                    "amortized_stream_bytes": amortized_bytes,
                    "amortized_stream_bpp": 8.0 * amortized_bytes / input_points,
                    "amortized_stream_note": ("accounting_only_not_a_decodable_stream"),
                }
            )
        enriched_gpcc.append(enriched)
    return {
        "source": {
            "experiment_dir": str(experiment_dir),
            "benchmark_metrics_sha256": file_sha256(metrics_path),
            "gpcc_per_cloud_sha256": file_sha256(gpcc_path),
            "constellation_per_cloud_sha256": file_sha256(constellation_path),
        },
        "protocol": {
            "input_points": input_points,
            "full_rate": "complete serialized stream bits / input points",
            "payload_rate": "byte-aligned payload bits / input points",
            "amortized_rate": (
                "SPS/GPS accounting only; not a decodable per-cloud stream"
                if amortize_parameter_sets_over is not None
                else None
            ),
        },
        "gpcc_summary": _gpcc_summary(enriched_gpcc),
        "constellation_summary": _constellation_summary(
            _jsonl(constellation_path), input_points=input_points
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--amortize-parameter-sets-over", type=int)
    args = parser.parse_args()
    result = resummarize(
        args.experiment_dir,
        amortize_parameter_sets_over=args.amortize_parameter_sets_over,
    )
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
