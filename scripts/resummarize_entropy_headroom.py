#!/usr/bin/env python3
"""Reconstruct Experiment 019 messages and summarize entropy-coding headroom."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.bitstream import (  # noqa: E402
    MODE_ENTROPY,
    ConstellationPacket,
    decode_constellation,
    encode_constellation,
    entropy_bound_bytes,
)
from pointconstellation.data import file_sha256  # noqa: E402
from pointconstellation.official_stability import (  # noqa: E402
    OfficialStabilityConfig,
    _load_models,
    _row_key,
)
from pointconstellation.refiner_experiment import _fps, _state_hash  # noqa: E402
from pointconstellation.stability_experiment import (  # noqa: E402
    StabilityExperimentConfig,
    _batch_metadata,
    _data_protocol,
    _datasets,
    _loader,
    _serialized_coordinates,
    _source,
)
from pointconstellation.train import select_device  # noqa: E402


def _read_rows(path: Path, split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"official per-cloud rows are missing: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"official rows do not contain split={split}")
    keys = [_row_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError(f"official rows contain duplicate keys for split={split}")
    return selected


def _packet_accounting(
    packet: ConstellationPacket, num_points: int
) -> dict[str, float | int]:
    entropy_stream = encode_constellation(
        packet.coordinates,
        bits=packet.bits,
        mode=MODE_ENTROPY,
        output_points=packet.output_points,
    )
    decoded = decode_constellation(entropy_stream)
    if not np.array_equal(packet.coordinates, decoded.coordinates):
        raise RuntimeError(
            "entropy stream changed the regenerated fixed-stream lattice"
        )
    entropy_bytes = len(entropy_stream)
    return {
        "stream_bytes": packet.stream_bytes,
        "actual_stream_bpp": 8.0 * packet.stream_bytes / num_points,
        "entropy_stream_bytes": entropy_bytes,
        "entropy_bpp": 8.0 * entropy_bytes / num_points,
        "entropy_bound_bytes": entropy_bound_bytes(
            packet.coordinates, bits=packet.bits
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = np.asarray([row["stream_bytes"] for row in rows], dtype=np.float64)
    entropy = np.asarray(
        [row["entropy_stream_bytes"] for row in rows], dtype=np.float64
    )
    bound = np.asarray([row["entropy_bound_bytes"] for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "mean_fixed_stream_bytes": float(fixed.mean()),
        "mean_entropy_stream_bytes": float(entropy.mean()),
        "median_entropy_stream_bytes": float(np.median(entropy)),
        "minimum_entropy_stream_bytes": int(entropy.min()),
        "maximum_entropy_stream_bytes": int(entropy.max()),
        "mean_entropy_bound_bytes": float(bound.mean()),
        "mean_rice_gap_over_oracle_bytes": float((entropy - bound).mean()),
        "mean_headroom_bytes": float((fixed - entropy).mean()),
        "mean_headroom_percent_of_fixed": float(
            100.0 * (fixed.mean() - entropy.mean()) / fixed.mean()
        ),
    }


def resummarize_entropy_headroom(
    config: OfficialStabilityConfig,
    *,
    official_rows_path: Path,
    split: str,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Regenerate selected messages without rerunning reconstruction metrics."""

    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    rows = _read_rows(official_rows_path, split)
    methods = {row["method"] for row in rows}
    if not methods <= {"fps", "refiner"}:
        raise ValueError(f"unsupported official methods: {sorted(methods)}")
    if {row["coordinate_bits"] for row in rows} != {stability.coordinate_bits}:
        raise ValueError(
            "official rows do not match the stability coordinate precision"
        )
    if {row["constellation_size"] for row in rows} != {stability.constellation_size}:
        raise ValueError("official rows do not match the stability constellation size")

    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    dataset = datasets[split]
    requested_clouds = {
        (row["family"], row["model_id"], row["sample_id"]) for row in rows
    }
    device = select_device(device_name)
    started = time.perf_counter()
    fps_accounting: dict[tuple[str, str, int], dict[str, float | int]] = {}
    refiner_accounting: dict[tuple[Any, ...], dict[str, float | int]] = {}

    for batch in _loader(dataset, config=stability, shuffle=False):
        source = _source(batch, device)
        coordinates = _fps(
            source, stability.constellation_size, stability.coordinate_bits
        )
        _, packets, exact, lattice_exact = _serialized_coordinates(
            coordinates, config=stability
        )
        if not exact or not lattice_exact:
            raise RuntimeError("regenerated FPS message failed fixed-stream checks")
        for index, packet in enumerate(packets):
            metadata = _batch_metadata(batch, index)
            cloud_key = (
                metadata["family"],
                metadata["model_id"],
                metadata["sample_id"],
            )
            if cloud_key in requested_clouds:
                fps_accounting[cloud_key] = _packet_accounting(
                    packet, stability.num_points
                )

    model_records = []
    refiner_cells = sorted(
        {
            (int(row["decoder_seed"]), int(row["refiner_seed"]))
            for row in rows
            if row["method"] == "refiner"
        }
    )
    for decoder_seed, refiner_seed in refiner_cells:
        decoder, refiner, metadata = _load_models(
            stability,
            config,
            decoder_seed=decoder_seed,
            refiner_seed=refiner_seed,
            device=device,
        )
        assert refiner is not None
        for batch in _loader(dataset, config=stability, shuffle=False):
            source = _source(batch, device)
            coordinates = refiner(
                source,
                stability.constellation_size,
                decoder=decoder,
                target=source,
                num_output_points=stability.num_points,
            )
            _, packets, exact, lattice_exact = _serialized_coordinates(
                coordinates, config=stability
            )
            if not exact or not lattice_exact:
                raise RuntimeError(
                    "regenerated refiner message failed fixed-stream checks"
                )
            for index, packet in enumerate(packets):
                cloud = _batch_metadata(batch, index)
                cloud_key = (
                    cloud["family"],
                    cloud["model_id"],
                    cloud["sample_id"],
                )
                if cloud_key in requested_clouds:
                    key = (decoder_seed, refiner_seed, *cloud_key)
                    refiner_accounting[key] = _packet_accounting(
                        packet, stability.num_points
                    )
        if _state_hash(decoder) != metadata["decoder_state_hash"]:
            raise RuntimeError("decoder changed during entropy resummarization")
        model_records.append(
            {
                "decoder_seed": decoder_seed,
                "refiner_seed": refiner_seed,
                **metadata,
                "decoder_unchanged_during_resummarization": True,
            }
        )

    accounting_rows = []
    for row in rows:
        cloud_key = (row["family"], row["model_id"], row["sample_id"])
        if row["method"] == "fps":
            accounting = fps_accounting.get(cloud_key)
        else:
            accounting = refiner_accounting.get(
                (
                    row["decoder_seed"],
                    row["refiner_seed"],
                    *cloud_key,
                )
            )
        if accounting is None:
            raise RuntimeError(f"failed to regenerate official row: {_row_key(row)}")
        if accounting["stream_bytes"] != row["stream_bytes"]:
            raise RuntimeError(
                "regenerated fixed stream differs from recorded byte count"
            )
        accounting_rows.append(
            {
                "split": row["split"],
                "method": row["method"],
                "decoder_seed": row["decoder_seed"],
                "refiner_seed": row.get("refiner_seed"),
                "family": row["family"],
                "model_id": row["model_id"],
                "sample_id": row["sample_id"],
                **accounting,
            }
        )

    by_method = {
        method: _summary([row for row in accounting_rows if row["method"] == method])
        for method in sorted(methods)
    }
    return {
        "experiment": "019_entropy_headroom_resummary",
        "scope": (
            "sealed stabilized Experiment 019 messages selected by Experiment 020 "
            "official per-cloud rows; no reconstruction metrics recomputed"
        ),
        "split": split,
        "device": str(device),
        "source": {
            "official_rows": str(official_rows_path),
            "official_rows_sha256": file_sha256(official_rows_path),
            "stability_config": str(stability_path),
            "stability_config_sha256": file_sha256(stability_path),
        },
        "data_protocol": data_protocol,
        "rows": len(accounting_rows),
        "summary": {
            "all": _summary(accounting_rows),
            "by_method": by_method,
        },
        "model_records": model_records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official-rows", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = resummarize_entropy_headroom(
        OfficialStabilityConfig.from_json(args.config),
        official_rows_path=args.official_rows,
        split=args.split,
        device_name=args.device,
    )
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
