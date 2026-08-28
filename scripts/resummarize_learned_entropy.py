#!/usr/bin/env python3
"""Fit mode-2 candidates on Experiment 019 train and score validation bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.bitstream import (  # noqa: E402
    MODE_ENTROPY,
    MODE_LEARNED,
    ConstellationPacket,
    decode_constellation,
    encode_constellation,
    entropy_bound_bytes,
)
from pointconstellation.data import file_sha256  # noqa: E402
from pointconstellation.learned_entropy import (  # noqa: E402
    AUTOREGRESSIVE,
    OCTREE,
    LearnedEntropyConfig,
    LearnedEntropyModel,
    fit_learned_entropy_model,
)
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


def _read_validation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"official per-cloud rows are missing: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == "validation"]
    if not selected:
        raise ValueError("official rows do not contain split=validation")
    keys = [_row_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("official validation rows contain duplicate keys")
    return selected


def _packet_lattice(packet: ConstellationPacket) -> np.ndarray:
    levels = (1 << packet.bits) - 1
    return np.rint((packet.normalized_coordinates + 1.0) * 0.5 * levels).astype(
        np.uint32
    )


def _packets_for_dataset(
    dataset: Any,
    *,
    stability: StabilityExperimentConfig,
    device: Any,
    decoder: Any | None = None,
    refiner: Any | None = None,
) -> dict[tuple[str, str, int], ConstellationPacket]:
    packets_by_cloud = {}
    for batch in _loader(dataset, config=stability, shuffle=False):
        source = _source(batch, device)
        if refiner is None:
            coordinates = _fps(
                source, stability.constellation_size, stability.coordinate_bits
            )
        else:
            coordinates = refiner(
                source,
                stability.constellation_size,
                decoder=decoder,
                target=source,
                num_output_points=stability.num_points,
            )
        _, packets, exact, lattice_exact = _serialized_coordinates(
            coordinates,
            config=stability,
            normalization_centers=batch.get("normalization_center"),
            normalization_scales=batch.get("normalization_scale"),
        )
        if not exact or not lattice_exact:
            raise RuntimeError("regenerated message failed fixed-stream checks")
        for index, packet in enumerate(packets):
            metadata = _batch_metadata(batch, index)
            key = (
                metadata["family"],
                metadata["model_id"],
                metadata["sample_id"],
            )
            packets_by_cloud[key] = packet
    return packets_by_cloud


def _candidate_accounting(
    packet: ConstellationPacket,
    model: LearnedEntropyModel,
) -> dict[str, float | int | bool]:
    stream = encode_constellation(
        packet.normalized_coordinates,
        bits=packet.bits,
        mode=MODE_LEARNED,
        output_points=packet.output_points,
        normalization_center=packet.normalization_center,
        normalization_scale=packet.normalization_scale,
        learned_model=model,
    )
    decoded = decode_constellation(stream, learned_model=model)
    if not np.array_equal(packet.coordinates, decoded.coordinates):
        raise RuntimeError("learned mode changed the regenerated fixed-stream lattice")
    bound = (
        entropy_bound_bytes(packet.normalized_coordinates, bits=packet.bits)
        + packet.normalization_bytes
    )
    return {
        "stream_bytes": len(stream),
        "entropy_bound_bytes": bound,
        "not_below_entropy_bound": len(stream) >= math.ceil(bound),
    }


def _summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = np.asarray([row["fixed_stream_bytes"] for row in values])
    rice = np.asarray([row["entropy_stream_bytes"] for row in values])
    learned = np.asarray([row["learned_stream_bytes"] for row in values])
    return {
        "rows": len(values),
        "mean_fixed_stream_bytes": float(fixed.mean()),
        "mean_mode_1_stream_bytes": float(rice.mean()),
        "mean_mode_2_stream_bytes": float(learned.mean()),
        "median_mode_2_stream_bytes": float(np.median(learned)),
        "minimum_mode_2_stream_bytes": int(learned.min()),
        "maximum_mode_2_stream_bytes": int(learned.max()),
        "mode_2_fraction_of_fixed": float(learned.mean() / fixed.mean()),
        "gate_g_a2_at_most_40_bytes": bool(learned.mean() <= 40.0),
        "all_mode_2_not_below_entropy_bound": all(
            row["not_below_entropy_bound"] for row in values
        ),
    }


def _rice_bytes(packet: ConstellationPacket) -> int:
    stream = encode_constellation(
        packet.normalized_coordinates,
        bits=packet.bits,
        mode=MODE_ENTROPY,
        output_points=packet.output_points,
        normalization_center=packet.normalization_center,
        normalization_scale=packet.normalization_scale,
    )
    decoded = decode_constellation(stream)
    if not np.array_equal(packet.coordinates, decoded.coordinates):
        raise RuntimeError("mode 1 changed the regenerated fixed-stream lattice")
    return len(stream)


def resummarize_learned_entropy(
    config: OfficialStabilityConfig,
    *,
    official_rows_path: Path,
    device_name: str | None = None,
    inference_batch_size: int = 32,
    max_refiner_cells: int | None = None,
    model_output: Path | None = None,
) -> dict[str, Any]:
    """Fit both candidates on train, select by validation bytes, and summarize."""

    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    if max_refiner_cells is not None and max_refiner_cells < 1:
        raise ValueError("max_refiner_cells must be positive")
    started = time.perf_counter()
    stability_path = Path(config.stability_config)
    base_stability = StabilityExperimentConfig.from_json(stability_path)
    stability = replace(base_stability, batch_size=inference_batch_size)
    rows = _read_validation_rows(official_rows_path)
    if {row["coordinate_bits"] for row in rows} != {stability.coordinate_bits}:
        raise ValueError("official rows do not match coordinate precision")
    if {row["constellation_size"] for row in rows} != {stability.constellation_size}:
        raise ValueError("official rows do not match constellation size")
    datasets = _datasets(stability)
    protocol = _data_protocol(stability, datasets)
    device = select_device(device_name)
    cells = sorted(
        {
            (int(row["decoder_seed"]), int(row["refiner_seed"]))
            for row in rows
            if row["method"] == "refiner"
        }
    )
    if max_refiner_cells is not None:
        cells = cells[:max_refiner_cells]
        selected_cells = set(cells)
        selected_decoders = {decoder_seed for decoder_seed, _ in cells}
        rows = [
            row
            for row in rows
            if (
                row["method"] == "fps" and int(row["decoder_seed"]) in selected_decoders
            )
            or (
                row["method"] == "refiner"
                and (int(row["decoder_seed"]), int(row["refiner_seed"]))
                in selected_cells
            )
        ]

    train_lattices = []
    fps_train = _packets_for_dataset(
        datasets["train"], stability=stability, device=device
    )
    train_lattices.extend(_packet_lattice(packet) for packet in fps_train.values())
    validation_packets: dict[tuple[Any, ...], ConstellationPacket] = {}
    fps_validation = _packets_for_dataset(
        datasets["validation"], stability=stability, device=device
    )
    model_records = []
    for decoder_seed, refiner_seed in cells:
        decoder, refiner, metadata = _load_models(
            stability,
            config,
            decoder_seed=decoder_seed,
            refiner_seed=refiner_seed,
            device=device,
        )
        assert refiner is not None
        train_packets = _packets_for_dataset(
            datasets["train"],
            stability=stability,
            device=device,
            decoder=decoder,
            refiner=refiner,
        )
        train_lattices.extend(
            _packet_lattice(packet) for packet in train_packets.values()
        )
        cell_validation = _packets_for_dataset(
            datasets["validation"],
            stability=stability,
            device=device,
            decoder=decoder,
            refiner=refiner,
        )
        for cloud_key, packet in cell_validation.items():
            validation_packets[(decoder_seed, refiner_seed, *cloud_key)] = packet
        if _state_hash(decoder) != metadata["decoder_state_hash"]:
            raise RuntimeError("decoder changed during learned entropy resummary")
        model_records.append(
            {
                "decoder_seed": decoder_seed,
                "refiner_seed": refiner_seed,
                **metadata,
                "decoder_unchanged_during_resummarization": True,
            }
        )

    training = np.stack(train_lattices)
    candidates = {
        candidate: fit_learned_entropy_model(
            training,
            bits=stability.coordinate_bits,
            config=LearnedEntropyConfig(candidate=candidate),
            training_split="train",
        )
        for candidate in (OCTREE, AUTOREGRESSIVE)
    }
    candidate_rows: dict[str, list[dict[str, Any]]] = {
        candidate: [] for candidate in candidates
    }
    for row in rows:
        cloud_key = (row["family"], row["model_id"], row["sample_id"])
        if row["method"] == "fps":
            packet = fps_validation.get(cloud_key)
        else:
            packet = validation_packets.get(
                (row["decoder_seed"], row["refiner_seed"], *cloud_key)
            )
        if packet is None:
            raise RuntimeError(f"failed to regenerate official row: {_row_key(row)}")
        if packet.stream_bytes != row["stream_bytes"]:
            raise RuntimeError("regenerated fixed stream byte count changed")
        base = {
            "split": row["split"],
            "method": row["method"],
            "decoder_seed": row["decoder_seed"],
            "refiner_seed": row.get("refiner_seed"),
            "family": row["family"],
            "model_id": row["model_id"],
            "sample_id": row["sample_id"],
            "constellation_size": packet.coordinates.shape[0],
            "coordinate_bits": packet.bits,
            "stream_bytes": packet.stream_bytes,
            "actual_stream_bpp": 8.0 * packet.stream_bytes / stability.num_points,
            "fixed_stream_bytes": packet.stream_bytes,
            "entropy_stream_bytes": _rice_bytes(packet),
        }
        base["entropy_bpp"] = 8.0 * base["entropy_stream_bytes"] / stability.num_points
        for candidate, model in candidates.items():
            accounting = _candidate_accounting(packet, model)
            candidate_rows[candidate].append(
                {
                    **base,
                    "learned_stream_bytes": accounting["stream_bytes"],
                    "learned_bpp": 8.0
                    * accounting["stream_bytes"]
                    / stability.num_points,
                    "entropy_bound_bytes": accounting["entropy_bound_bytes"],
                    "not_below_entropy_bound": accounting["not_below_entropy_bound"],
                }
            )

    candidate_summary = {
        candidate: {
            **_summarize(candidate_rows[candidate]),
            "model_hash": model.model_hash,
            "shared_model_parameter_bytes": model.parameter_bytes,
            "shared_model_serialized_bytes": len(model.to_bytes()),
            "training_streams": model.training_streams,
            "training_split": model.training_split,
            "config": asdict(model.config),
        }
        for candidate, model in candidates.items()
    }
    selected_name = min(
        candidates,
        key=lambda name: (
            candidate_summary[name]["mean_mode_2_stream_bytes"],
            name,
        ),
    )
    selected_model = candidates[selected_name]
    selected_rows = [
        {
            **row,
            "learned_model_candidate": selected_name,
            "learned_model_hash": selected_model.model_hash,
            "learned_shared_model_parameter_bytes": selected_model.parameter_bytes,
        }
        for row in candidate_rows[selected_name]
    ]
    if model_output is not None:
        selected_model.save(model_output)
        serialized_model_bytes = model_output.stat().st_size
        if (
            file_sha256(model_output)
            != hashlib.sha256(selected_model.to_bytes()).hexdigest()
        ):
            raise RuntimeError("saved learned entropy model bytes changed")
    else:
        serialized_model_bytes = len(selected_model.to_bytes())

    by_method = {
        method: _summarize([row for row in selected_rows if row["method"] == method])
        for method in sorted({row["method"] for row in selected_rows})
    }
    return {
        "experiment": "019_learned_entropy_resummary",
        "scope": (
            "shared mode-2 models fitted only to regenerated Experiment 019 "
            "training constellations; candidate selected by official validation bytes"
        ),
        "declared_stream_mode": 0,
        "candidate_selection_split": "validation",
        "model_training_split": "train",
        "complete_factorial": max_refiner_cells is None,
        "refiner_cells": len(cells),
        "device": str(device),
        "source": {
            "official_rows": str(official_rows_path),
            "official_rows_sha256": file_sha256(official_rows_path),
            "stability_config": str(stability_path),
            "stability_config_sha256": file_sha256(stability_path),
        },
        "data_protocol": protocol,
        "training_streams": len(training),
        "validation_rows": len(rows),
        "candidate_summary": candidate_summary,
        "selected_candidate": selected_name,
        "selected_model_hash": selected_model.model_hash,
        "selected_shared_model_parameter_bytes": selected_model.parameter_bytes,
        "selected_shared_model_serialized_bytes": serialized_model_bytes,
        "summary": {
            "all": _summarize(selected_rows),
            "by_method": by_method,
        },
        "per_cloud": selected_rows,
        "model_records": model_records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official-rows", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument(
        "--max-refiner-cells",
        type=int,
        help="explicit smoke limit; omit for the complete factorial",
    )
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = resummarize_learned_entropy(
        OfficialStabilityConfig.from_json(args.config),
        official_rows_path=args.official_rows,
        device_name=args.device,
        inference_batch_size=args.inference_batch_size,
        max_refiner_cells=args.max_refiner_cells,
        model_output=args.model_output,
    )
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
