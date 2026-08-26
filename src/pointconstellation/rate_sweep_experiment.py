"""Experiment 025: stabilized multi-rate Adam/STE constellation sweep.

The encoder-visible source sample is the only optimization target.  Every
evaluated message is independently serialized as an unordered, quantized
``K x 3`` coordinate set before the frozen stabilized decoder sees it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from pointconstellation.bitstream import (
    HEADER,
    decode_constellation,
    encode_constellation,
    expected_payload_bytes,
    expected_stream_bytes,
)
from pointconstellation.codecs import parse_gpcc_stream, run_pc_error
from pointconstellation.data import file_sha256
from pointconstellation.headroom_experiment import (
    _batch_tensors,
    _initial_starts,
    _metadata,
    _search_adam_start,
    _select_best_by_source_score,
    _source_scorer,
)
from pointconstellation.official_stability import (
    OfficialStabilityConfig,
    _bootstrap_comparison,
    _load_models,
    _synchronize,
)
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _per_cloud_chamfer,
)
from pointconstellation.train import select_device

SUPPORTED_SPLITS = ("validation", "ood")
METHODS = ("fps", "adam_ste", "adam_multistart", "refiner")
PRIMARY_METHOD = "adam_ste"
SUMMARY_METRICS = ("source_chamfer_mse", "fresh_chamfer_mse", "d1_mse", "d2_mse")


@dataclass(frozen=True)
class RateSweepExperimentConfig:
    """Validated multi-rate configuration over sealed stabilized decoders."""

    stability_config: str = "configs/experiment_019_stability_modelnet40.json"
    stability_artifact_dir: str = "artifacts/local/experiment_019_stability_modelnet40"
    pc_error_executable: str = "artifacts/tools/mpeg-pcc-dmetric/build/Release/pc_error"
    gpcc_reference_path: str | None = None
    position_bits: int = 12
    timeout_seconds: float = 120.0
    constellation_sizes: tuple[int, ...] = (4, 6, 8, 12, 16)
    coordinate_bits: tuple[int, ...] = (8, 10, 12)
    decoder_seeds: tuple[int, ...] = (7, 17, 29, 41, 53, 67)
    refiner_seeds: tuple[int, ...] = (101, 211, 307)
    refiner_constellation_size: int = 8
    start_methods: tuple[str, ...] = ("fps", "kmeans")
    random_start_seeds: tuple[int, ...] = (101, 211)
    adam_evaluations: int = 64
    adam_learning_rate: float = 0.03
    selection_seed: int = 20_260_825
    splits: tuple[str, ...] = SUPPORTED_SPLITS
    max_clouds_per_split: int | None = None
    batch_size: int = 4
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_825
    confidence_level: float = 0.95
    output_dir: str = "artifacts/local/experiment_025_rate_sweep_modelnet40"

    def __post_init__(self) -> None:
        if not 2 <= self.position_bits <= 24:
            raise ValueError("position_bits must be between 2 and 24")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (
            not self.constellation_sizes
            or len(set(self.constellation_sizes)) != len(self.constellation_sizes)
            or tuple(sorted(self.constellation_sizes)) != self.constellation_sizes
            or min(self.constellation_sizes) < 2
        ):
            raise ValueError(
                "constellation_sizes must be unique, increasing, and at least two"
            )
        if (
            not self.coordinate_bits
            or len(set(self.coordinate_bits)) != len(self.coordinate_bits)
            or tuple(sorted(self.coordinate_bits)) != self.coordinate_bits
            or min(self.coordinate_bits) < 2
            or max(self.coordinate_bits) > 24
        ):
            raise ValueError(
                "coordinate_bits must be unique, increasing values from 2 to 24"
            )
        if not self.decoder_seeds or len(set(self.decoder_seeds)) != len(
            self.decoder_seeds
        ):
            raise ValueError("decoder_seeds must be nonempty and unique")
        if len(set(self.refiner_seeds)) != len(self.refiner_seeds):
            raise ValueError("refiner_seeds must be unique")
        if self.refiner_constellation_size not in self.constellation_sizes:
            raise ValueError("refiner_constellation_size must be in the K grid")
        if set(self.start_methods) != {"fps", "kmeans"} or len(self.start_methods) != 2:
            raise ValueError("start_methods must contain fps and kmeans exactly once")
        if (
            len(self.random_start_seeds) != 2
            or len(set(self.random_start_seeds)) != 2
            or min(self.random_start_seeds) < 0
        ):
            raise ValueError(
                "random_start_seeds must contain two unique nonnegative seeds"
            )
        if self.adam_evaluations < 1 or self.adam_learning_rate <= 0:
            raise ValueError("Adam evaluations and learning rate must be positive")
        if not 0 <= self.selection_seed < 2**63:
            raise ValueError("selection_seed must be a nonnegative 63-bit integer")
        if (
            not self.splits
            or len(set(self.splits)) != len(self.splits)
            or set(self.splits) - set(SUPPORTED_SPLITS)
        ):
            raise ValueError("splits must be unique validation and/or ood entries")
        if self.max_clouds_per_split is not None and self.max_clouds_per_split < 1:
            raise ValueError("max_clouds_per_split must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

    @classmethod
    def from_json(cls, path: Path) -> RateSweepExperimentConfig:
        values = json.loads(path.read_text())
        for key in (
            "constellation_sizes",
            "coordinate_bits",
            "decoder_seeds",
            "refiner_seeds",
            "start_methods",
            "random_start_seeds",
            "splits",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @property
    def start_labels(self) -> tuple[str, ...]:
        """The four predeclared starts used by the multi-start bound."""

        return (
            *self.start_methods,
            *(f"random_seed_{seed}" for seed in self.random_start_seeds),
        )


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _cell_id(constellation_size: int, coordinate_bits: int) -> str:
    return f"k_{constellation_size}_q_{coordinate_bits}"


def _cells(config: RateSweepExperimentConfig) -> tuple[tuple[int, int], ...]:
    return tuple(
        (constellation_size, bits)
        for constellation_size in config.constellation_sizes
        for bits in config.coordinate_bits
    )


def _rate_fields(
    constellation_size: int, coordinate_bits: int, input_points: int
) -> dict[str, int | float]:
    payload_bytes = expected_payload_bytes(constellation_size, coordinate_bits)
    stream_bytes = expected_stream_bytes(constellation_size, coordinate_bits)
    return {
        "header_bytes": HEADER.size,
        "payload_bytes": payload_bytes,
        "stream_bytes": stream_bytes,
        "payload_bpp": 8.0 * payload_bytes / input_points,
        "actual_stream_bpp": 8.0 * stream_bytes / input_points,
    }


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["split"],
        row["method"],
        row["decoder_seed"],
        row.get("refiner_seed"),
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
        raise RuntimeError(f"rate-sweep cell contains duplicate rows: {path}")
    return rows


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    temporary.replace(path)


def _write_manifest_once(path: Path, value: Mapping[str, Any]) -> None:
    serialized = json.dumps(value, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if json.loads(path.read_text()) != value:
            raise RuntimeError(
                f"existing rate-sweep manifest differs: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _official_config(
    config: RateSweepExperimentConfig,
    stability: StabilityExperimentConfig,
) -> OfficialStabilityConfig:
    return OfficialStabilityConfig(
        stability_config=config.stability_config,
        stability_artifact_dir=config.stability_artifact_dir,
        pc_error_executable=config.pc_error_executable,
        position_bits=config.position_bits,
        timeout_seconds=config.timeout_seconds,
        decoder_seeds=stability.decoder_seeds,
        refiner_seeds=stability.refiner_seeds,
        splits=config.splits,
        max_clouds_per_split=(
            config.max_clouds_per_split
            if config.max_clouds_per_split is None or config.max_clouds_per_split >= 2
            else 2
        ),
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
        output_dir=config.output_dir,
    )


def _serialize_coordinates(
    coordinates: Tensor,
    *,
    bits: int,
    mode: str,
    output_points: int,
) -> tuple[Tensor, list[bytes], bool, bool, float]:
    started = time.perf_counter()
    decoded = []
    streams = []
    round_trip_exact = True
    lattice_exact = True
    for coordinate_row in coordinates.detach().cpu().numpy():
        stream = encode_constellation(
            coordinate_row,
            bits=bits,
            mode=mode,
            output_points=output_points,
        )
        packet = decode_constellation(stream)
        levels = (1 << packet.bits) - 1
        lattice = (packet.coordinates + 1.0) * 0.5 * levels
        lattice_exact = lattice_exact and bool(
            np.all(np.abs(lattice - np.rint(lattice)) <= 1e-9)
        )
        round_trip_exact = round_trip_exact and (
            encode_constellation(
                packet.coordinates,
                bits=packet.bits,
                mode=packet.mode,
                output_points=packet.output_points,
            )
            == stream
        )
        streams.append(stream)
        decoded.append(torch.from_numpy(packet.coordinates).float())
    return (
        torch.stack(decoded).to(coordinates.device),
        streams,
        round_trip_exact,
        lattice_exact,
        time.perf_counter() - started,
    )


def _metric_cache_key(
    *,
    decoder_seed: int,
    split: str,
    metadata: Mapping[str, Any],
    stream_sha256: str,
) -> tuple[Any, ...]:
    return (
        decoder_seed,
        split,
        metadata["family"],
        metadata["model_id"],
        metadata["sample_id"],
        stream_sha256,
    )


def _metric_cache(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = _metric_cache_key(
            decoder_seed=int(row["decoder_seed"]),
            split=str(row["split"]),
            metadata=row,
            stream_sha256=str(row["stream_sha256"]),
        )
        metrics = {
            name: row[name]
            for name in row
            if name.startswith("d1_") or name.startswith("d2_")
        }
        metrics["official_metric_seconds"] = row["official_metric_seconds"]
        if key in result and result[key] != metrics:
            raise RuntimeError("identical streams have inconsistent official metrics")
        result[key] = metrics
    return result


def _evaluate_coordinates(
    coordinates: Tensor,
    samples: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    *,
    decoder: nn.Module,
    stability: StabilityExperimentConfig,
    config: RateSweepExperimentConfig,
    constellation_size: int,
    coordinate_bits: int,
    dataset_name: str,
    decoder_seed: int,
    split: str,
    method_fields: Sequence[Mapping[str, Any]],
    encode_seconds: Sequence[float],
    scratch_root: Path,
    metric_cache: dict[tuple[Any, ...], dict[str, Any]],
    mode: str = "free",
) -> list[dict[str, Any]]:
    if not (
        len(coordinates)
        == len(samples)
        == len(metadata)
        == len(method_fields)
        == len(encode_seconds)
    ):
        raise ValueError("coordinate evaluation batches must align")
    source, fresh, normals = _batch_tensors(samples, coordinates.device)
    decoded, streams, exact, lattice_exact, serialization_seconds = (
        _serialize_coordinates(
            coordinates,
            bits=coordinate_bits,
            mode=mode,
            output_points=stability.num_points,
        )
    )
    _synchronize(coordinates.device)
    decode_started = time.perf_counter()
    with torch.no_grad():
        reconstruction = decoder(decoded, num_output_points=stability.num_points)
        source_losses = _per_cloud_chamfer(
            reconstruction,
            source,
            chunk_size=stability.distance_chunk_size,
        )
        fresh_losses = _per_cloud_chamfer(
            reconstruction,
            fresh,
            chunk_size=stability.distance_chunk_size,
        )
    _synchronize(coordinates.device)
    decode_per_cloud = (time.perf_counter() - decode_started) / len(samples)
    serialization_per_cloud = serialization_seconds / len(samples)
    rate = _rate_fields(constellation_size, coordinate_bits, stability.num_points)
    rows = []
    for index, (cloud_metadata, fields, stream) in enumerate(
        zip(metadata, method_fields, streams, strict=True)
    ):
        stream_sha256 = hashlib.sha256(stream).hexdigest()
        cache_key = _metric_cache_key(
            decoder_seed=decoder_seed,
            split=split,
            metadata=cloud_metadata,
            stream_sha256=stream_sha256,
        )
        if cache_key not in metric_cache:
            with tempfile.TemporaryDirectory(
                prefix=f"{_cell_id(constellation_size, coordinate_bits)}-{split}-",
                dir=scratch_root,
            ) as temporary:
                official = run_pc_error(
                    Path(config.pc_error_executable),
                    source[index].detach().cpu().numpy(),
                    reconstruction[index].detach().cpu().numpy(),
                    normals[index].detach().cpu().numpy(),
                    work_dir=Path(temporary),
                    position_bits=config.position_bits,
                    timeout_seconds=config.timeout_seconds,
                )
            if not {"d1_mse", "d2_mse"} <= official.metrics.keys():
                raise RuntimeError("pc_error result is missing official D1 or D2 MSE")
            metric_cache[cache_key] = {
                **official.metrics,
                "official_metric_seconds": official.elapsed_seconds,
            }
        if len(stream) != rate["stream_bytes"]:
            raise RuntimeError("serialized stream differs from expected_stream_bytes")
        rows.append(
            {
                "experiment": "025_rate_sweep_adam",
                "dataset": dataset_name,
                "input_points": stability.num_points,
                "arm_label": "stabilized",
                "quality_device": coordinates.device.type,
                "rate_point": _cell_id(constellation_size, coordinate_bits),
                "split": split,
                "decoder_seed": decoder_seed,
                **cloud_metadata,
                **fields,
                "constellation_size": constellation_size,
                "coordinate_bits": coordinate_bits,
                "bitstream_mode": mode,
                "stream_hex": stream.hex(),
                "stream_sha256": stream_sha256,
                **rate,
                "source_chamfer_mse": float(source_losses[index].item()),
                "fresh_chamfer_mse": float(fresh_losses[index].item()),
                "source_only_optimization": True,
                "serialized_round_trip_exact": exact,
                "coordinates_on_exact_lattice": lattice_exact,
                "encode_seconds": float(encode_seconds[index])
                + serialization_per_cloud,
                "decode_seconds": decode_per_cloud,
                **metric_cache[cache_key],
            }
        )
    return rows


def _expected_methods(
    config: RateSweepExperimentConfig, constellation_size: int
) -> tuple[tuple[str, int | None], ...]:
    result: list[tuple[str, int | None]] = [
        ("fps", None),
        ("adam_ste", None),
        ("adam_multistart", None),
    ]
    if constellation_size == config.refiner_constellation_size:
        result.extend(("refiner", seed) for seed in config.refiner_seeds)
    return tuple(result)


def _expected_keys(
    config: RateSweepExperimentConfig,
    *,
    constellation_size: int,
    decoder_seed: int,
    split: str,
    metadata: Sequence[Mapping[str, Any]],
) -> set[tuple[Any, ...]]:
    return {
        (
            split,
            method,
            decoder_seed,
            refiner_seed,
            cloud["family"],
            cloud["model_id"],
            cloud["sample_id"],
        )
        for method, refiner_seed in _expected_methods(config, constellation_size)
        for cloud in metadata
    }


def _validate_resumed_rates(
    rows: Sequence[Mapping[str, Any]],
    *,
    constellation_size: int,
    coordinate_bits: int,
    input_points: int,
) -> None:
    expected = _rate_fields(constellation_size, coordinate_bits, input_points)
    fields = tuple(expected)
    for row in rows:
        if any(row.get(field) != expected[field] for field in fields):
            raise RuntimeError("resumed row differs from exact cell rate accounting")
        if (
            row.get("constellation_size") != constellation_size
            or row.get("coordinate_bits") != coordinate_bits
        ):
            raise RuntimeError("resumed row belongs to another rate cell")
        if len(bytes.fromhex(str(row["stream_hex"]))) != expected["stream_bytes"]:
            raise RuntimeError(
                "resumed stream hex differs from its declared byte count"
            )


def _comparison(
    baseline: np.ndarray,
    candidate: np.ndarray,
    categories: np.ndarray,
    *,
    baseline_method: str,
    candidate_method: str,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    result = _bootstrap_comparison(
        baseline,
        candidate,
        categories,
        samples=samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "decoder_count": result["decoder_count"],
        "candidate_replicate_count": result["refiner_count"],
        "cloud_count": result["cloud_count"],
        "category_count": result["category_count"],
        "bootstrap_samples": result["bootstrap_samples"],
        "confidence_level": result["confidence_level"],
        "aggregate_baseline_rmse": result["aggregate_fps_rmse_grid_units"],
        "aggregate_candidate_rmse": result["aggregate_refiner_rmse_grid_units"],
        "relative_rmse_improvement_percent": result[
            "relative_rmse_improvement_percent"
        ],
        "confidence_interval_lower_percent": result[
            "confidence_interval_lower_percent"
        ],
        "confidence_interval_upper_percent": result[
            "confidence_interval_upper_percent"
        ],
        "passes_positive_interval": result["passes_positive_interval"],
    }


def _reference_split(split: str) -> tuple[str, ...]:
    if split == "ood":
        return ("ood", "category_ood", "parameter_ood")
    return (split,)


def _load_gpcc_rows(path: Path | None, *, input_points: int) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"rate-accounted G-PCC reference is missing: {path}")
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        document = json.loads(path.read_text())
        rows = document.get("rows", document.get("per_cloud", []))
    result = []
    for row in rows:
        if row.get("method") not in {"gpcc", "gpcc_octree"}:
            continue
        payload_bytes = row.get("payload_bytes")
        payload_bpp = row.get("payload_bpp")
        if payload_bytes is None or payload_bpp is None:
            stream_path = (
                path.parent
                / "gpcc_work"
                / str(row["split"])
                / f"sample_{int(row['sample_id']):05d}"
                / str(row["rate_point"])
                / "stream.bin"
            )
            if not stream_path.is_file():
                raise RuntimeError(
                    "G-PCC reference lacks Experiment 021 payload fields and its "
                    f"source stream is unavailable: {stream_path}"
                )
            breakdown = parse_gpcc_stream(stream_path)
            if breakdown.total_bytes != row["stream_bytes"]:
                raise RuntimeError(
                    f"recorded G-PCC size differs from source stream: {stream_path}"
                )
            payload_bytes = breakdown.payload_bytes
            payload_bpp = 8.0 * payload_bytes / input_points
        stream_bytes = float(row["stream_bytes"])
        payload_bytes = float(payload_bytes)
        payload_bpp = float(payload_bpp)
        if not 0 <= payload_bytes <= stream_bytes:
            raise RuntimeError("G-PCC payload bytes are outside the complete stream")
        if not math.isclose(
            payload_bpp,
            8.0 * payload_bytes / input_points,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("G-PCC payload_bpp differs from byte-exact payload")
        actual_stream_bpp = row.get("actual_stream_bpp")
        if actual_stream_bpp is not None and not math.isclose(
            float(actual_stream_bpp),
            8.0 * stream_bytes / input_points,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("G-PCC actual_stream_bpp differs from stream bytes")
        d1_mse = row.get("d1_mse", row.get("official_d1_mse"))
        d2_mse = row.get("d2_mse", row.get("official_d2_mse"))
        if d1_mse is None or d2_mse is None:
            raise RuntimeError("G-PCC reference lacks official D1/D2 MSE")
        result.append(
            {
                **row,
                "header_bytes": stream_bytes - payload_bytes,
                "payload_bytes": payload_bytes,
                "payload_bpp": payload_bpp,
                "d1_mse": float(d1_mse),
                "d2_mse": float(d2_mse),
            }
        )
    if not result:
        raise RuntimeError("G-PCC reference contains no usable per-cloud rows")
    keys = [
        (
            row["split"],
            row["rate_point"],
            row["family"],
            row["model_id"],
            row["sample_id"],
        )
        for row in result
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("G-PCC reference contains duplicate per-cloud rows")
    return result


def _gpcc_reference_digest(rows: Sequence[Mapping[str, Any]]) -> str | None:
    if not rows:
        return None
    identity_fields = (
        "split",
        "rate_point",
        "family",
        "model_id",
        "sample_id",
        "stream_bytes",
        "header_bytes",
        "payload_bytes",
        "actual_stream_bpp",
        "payload_bpp",
        "d1_mse",
        "d2_mse",
    )
    identities = [
        {field: row.get(field) for field in identity_fields}
        for row in sorted(
            rows,
            key=lambda row: (
                row["split"],
                row["rate_point"],
                row["family"],
                row["model_id"],
                row["sample_id"],
            ),
        )
    ]
    serialized = json.dumps(identities, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _candidate_array(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    split: str,
    method: str,
    decoder_seeds: Sequence[int],
    refiner_seeds: Sequence[int | None],
    cloud_keys: Sequence[tuple[Any, ...]],
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            [
                [
                    indexed[(split, method, decoder_seed, refiner_seed, *cloud_key)][
                        metric
                    ]
                    for cloud_key in cloud_keys
                ]
                for refiner_seed in refiner_seeds
            ]
            for decoder_seed in decoder_seeds
        ],
        dtype=np.float64,
    )


def summarize_cell_rows(
    rows: list[dict[str, Any]],
    config: RateSweepExperimentConfig,
    *,
    constellation_size: int,
    coordinate_bits: int,
    gpcc_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build per-arm table rows and paired comparisons for one complete cell."""

    indexed = {_row_key(row): row for row in rows}
    comparisons = []
    table = []
    nearest_gpcc = []
    for split_index, split in enumerate(config.splits):
        cloud_keys = sorted(
            {
                (row["family"], row["model_id"], row["sample_id"])
                for row in rows
                if row["split"] == split
            }
        )
        if not cloud_keys:
            raise RuntimeError(f"rate-sweep cell has no rows for split={split}")
        categories = np.asarray([key[0] for key in cloud_keys])
        methods = ["fps", "adam_ste", "adam_multistart"]
        if constellation_size == config.refiner_constellation_size:
            methods.append("refiner")
        arrays = {}
        for method in methods:
            replicate_seeds: Sequence[int | None] = (
                config.refiner_seeds if method == "refiner" else (None,)
            )
            arrays[method] = {
                metric: _candidate_array(
                    indexed,
                    split=split,
                    method=method,
                    decoder_seeds=config.decoder_seeds,
                    refiner_seeds=replicate_seeds,
                    cloud_keys=cloud_keys,
                    metric=metric,
                )
                for metric in SUMMARY_METRICS
            }
            group = [
                row for row in rows if row["split"] == split and row["method"] == method
            ]
            first = group[0]
            table.append(
                {
                    "dataset": first["dataset"],
                    "split": split,
                    "method": method,
                    "arm_label": "stabilized",
                    "rate_point": first["rate_point"],
                    "constellation_size": constellation_size,
                    "coordinate_bits": coordinate_bits,
                    "header_bytes": first["header_bytes"],
                    "payload_bytes": first["payload_bytes"],
                    "stream_bytes": first["stream_bytes"],
                    "payload_bpp": first["payload_bpp"],
                    "actual_stream_bpp": first["actual_stream_bpp"],
                    "clouds": len(cloud_keys),
                    "decoder_seeds": len(config.decoder_seeds),
                    "replicates": len(replicate_seeds),
                    "decoder_evaluations": first["decoder_evaluations"],
                    **{
                        metric: float(arrays[method][metric].mean())
                        for metric in SUMMARY_METRICS
                    },
                    **{
                        metric.replace("_mse", "_rmse"): math.sqrt(
                            float(arrays[method][metric].mean())
                        )
                        for metric in SUMMARY_METRICS
                    },
                }
            )
        fps = arrays["fps"]
        for metric_index, metric in enumerate(("d1_mse", "d2_mse")):
            fps_baseline = fps[metric][:, 0, :]
            for method_index, method in enumerate(methods[1:]):
                comparisons.append(
                    {
                        "split": split,
                        "metric": metric,
                        "comparison_role": "method_vs_fps",
                        **_comparison(
                            fps_baseline,
                            arrays[method][metric],
                            categories,
                            baseline_method="fps",
                            candidate_method=method,
                            samples=config.bootstrap_samples,
                            confidence_level=config.confidence_level,
                            seed=(
                                config.bootstrap_seed
                                + 100_000 * constellation_size
                                + 1_000 * coordinate_bits
                                + 100 * split_index
                                + 10 * metric_index
                                + method_index
                            ),
                        ),
                    }
                )

        reference = [
            row for row in gpcc_rows if row["split"] in _reference_split(split)
        ]
        if not reference:
            continue
        by_rate: dict[str, list[Mapping[str, Any]]] = {}
        for row in reference:
            by_rate.setdefault(str(row["rate_point"]), []).append(row)
        target_payload = float(table[-1]["payload_bytes"])
        complete_rates = {
            rate_point: group
            for rate_point, group in by_rate.items()
            if {(row["family"], row["model_id"], row["sample_id"]) for row in group}
            == set(cloud_keys)
        }
        if not complete_rates:
            raise RuntimeError(
                f"G-PCC reference has no complete point for rate-sweep split={split}"
            )
        rate_point, gpcc_group = min(
            complete_rates.items(),
            key=lambda item: (
                abs(
                    float(np.mean([row["payload_bytes"] for row in item[1]]))
                    - target_payload
                ),
                item[0],
            ),
        )
        gpcc_index = {
            (row["family"], row["model_id"], row["sample_id"]): row
            for row in gpcc_group
        }
        gpcc_rate = {
            "split": split,
            "rate_point": rate_point,
            "mean_payload_bytes": float(
                np.mean([row["payload_bytes"] for row in gpcc_group])
            ),
            "mean_stream_bytes": float(
                np.mean([row["stream_bytes"] for row in gpcc_group])
            ),
            "mean_payload_bpp": float(
                np.mean([row["payload_bpp"] for row in gpcc_group])
            ),
        }
        nearest_gpcc.append(gpcc_rate)
        for metric_index, metric in enumerate(("d1_mse", "d2_mse")):
            gpcc_values = np.asarray(
                [gpcc_index[key][metric] for key in cloud_keys], dtype=np.float64
            )
            gpcc_baseline = np.broadcast_to(
                gpcc_values, (len(config.decoder_seeds), len(gpcc_values))
            )
            for method_index, method in enumerate(methods):
                comparisons.append(
                    {
                        "split": split,
                        "metric": metric,
                        "comparison_role": "method_vs_nearest_gpcc_payload_rate",
                        "nearest_gpcc": gpcc_rate,
                        **_comparison(
                            gpcc_baseline,
                            arrays[method][metric],
                            categories,
                            baseline_method="gpcc_octree",
                            candidate_method=method,
                            samples=config.bootstrap_samples,
                            confidence_level=config.confidence_level,
                            seed=(
                                config.bootstrap_seed
                                + 900_000
                                + 100_000 * constellation_size
                                + 1_000 * coordinate_bits
                                + 100 * split_index
                                + 10 * metric_index
                                + method_index
                            ),
                        ),
                    }
                )
    return {
        "cell_id": _cell_id(constellation_size, coordinate_bits),
        "constellation_size": constellation_size,
        "coordinate_bits": coordinate_bits,
        "rates": _rate_fields(
            constellation_size, coordinate_bits, int(rows[0]["input_points"])
        )
        if "input_points" in rows[0]
        else {
            key: rows[0][key]
            for key in (
                "header_bytes",
                "payload_bytes",
                "stream_bytes",
                "payload_bpp",
                "actual_stream_bpp",
            )
        },
        "table": table,
        "comparisons": comparisons,
        "nearest_gpcc": nearest_gpcc,
    }


def _cell_expected_row_count(
    config: RateSweepExperimentConfig,
    stability: StabilityExperimentConfig,
    datasets: Mapping[str, Any],
    constellation_size: int,
) -> int:
    clouds = sum(
        min(len(datasets[split]), config.max_clouds_per_split)
        if config.max_clouds_per_split is not None
        else len(datasets[split])
        for split in config.splits
    )
    return (
        clouds
        * len(config.decoder_seeds)
        * len(_expected_methods(config, constellation_size))
    )


def _run_cell(
    config: RateSweepExperimentConfig,
    stability: StabilityExperimentConfig,
    datasets: Mapping[str, Any],
    *,
    constellation_size: int,
    coordinate_bits: int,
    dataset_name: str,
    device: torch.device,
    gpcc_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    cell_dir = output_dir / "cells" / _cell_id(constellation_size, coordinate_bits)
    cell_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = cell_dir / "metric_scratch"
    scratch_root.mkdir(exist_ok=True)
    rate = _rate_fields(constellation_size, coordinate_bits, stability.num_points)
    _write_manifest_once(
        cell_dir / "cell_manifest.json",
        {
            "experiment": "025_rate_sweep_adam",
            "cell_id": _cell_id(constellation_size, coordinate_bits),
            "constellation_size": constellation_size,
            "coordinate_bits": coordinate_bits,
            "rate": rate,
            "decoder_protocol": "shared_experiment_019_variable_cardinality",
            "adam_evaluations_per_start": config.adam_evaluations,
            "multi_start_count": len(config.start_labels),
            "multi_start_total_evaluations": config.adam_evaluations
            * len(config.start_labels),
        },
    )
    rows_path = cell_dir / "rate_sweep_per_cloud.jsonl"
    rows = _load_rows(rows_path)
    resumed_rows = len(rows)
    _validate_resumed_rates(
        rows,
        constellation_size=constellation_size,
        coordinate_bits=coordinate_bits,
        input_points=stability.num_points,
    )
    completed = {_row_key(row) for row in rows}
    metric_cache = _metric_cache(rows)
    official = _official_config(config, stability)
    model_records = []
    started = time.perf_counter()
    for decoder_seed in config.decoder_seeds:
        decoder, _, decoder_metadata = _load_models(
            stability,
            official,
            decoder_seed=decoder_seed,
            refiner_seed=None,
            device=device,
        )
        decoder_hash = _state_hash(decoder)
        model_records.append(
            {
                "decoder_seed": decoder_seed,
                "refiner_seed": None,
                **decoder_metadata,
            }
        )
        refiners: dict[int, nn.Module] = {}
        if constellation_size == config.refiner_constellation_size:
            for refiner_seed in config.refiner_seeds:
                pair_decoder, refiner, metadata = _load_models(
                    stability,
                    official,
                    decoder_seed=decoder_seed,
                    refiner_seed=refiner_seed,
                    device=device,
                )
                assert refiner is not None
                if _state_hash(pair_decoder) != decoder_hash:
                    raise RuntimeError(
                        "refiner pair uses a different stabilized decoder"
                    )
                refiners[refiner_seed] = refiner
                model_records.append(
                    {
                        "decoder_seed": decoder_seed,
                        "refiner_seed": refiner_seed,
                        **metadata,
                    }
                )

        for split in config.splits:
            dataset = datasets[split]
            cloud_count = (
                len(dataset)
                if config.max_clouds_per_split is None
                else min(len(dataset), config.max_clouds_per_split)
            )
            for batch_start in range(0, cloud_count, config.batch_size):
                indices = range(
                    batch_start, min(batch_start + config.batch_size, cloud_count)
                )
                samples = [dataset[index] for index in indices]
                cloud_metadata = [
                    _metadata(sample, index)
                    for sample, index in zip(samples, indices, strict=True)
                ]
                expected_keys = _expected_keys(
                    config,
                    constellation_size=constellation_size,
                    decoder_seed=decoder_seed,
                    split=split,
                    metadata=cloud_metadata,
                )
                if expected_keys <= completed:
                    continue
                source, _, _ = _batch_tensors(samples, device)
                starts, selector_seconds = _initial_starts(
                    source,
                    cloud_metadata,
                    config,
                    split=split,
                    constellation_size=constellation_size,
                    bits=coordinate_bits,
                )
                scorer = _source_scorer(
                    decoder,
                    source,
                    num_output_points=stability.num_points,
                    chunk_size=stability.distance_chunk_size,
                )
                search_results = {}
                search_seconds = {}
                for label in config.start_labels:
                    _synchronize(device)
                    search_started = time.perf_counter()
                    result = _search_adam_start(
                        scorer,
                        starts[label],
                        bits=coordinate_bits,
                        budget=config.adam_evaluations,
                        learning_rate=config.adam_learning_rate,
                    )
                    _synchronize(device)
                    if result.decoder_evaluations_per_cloud != config.adam_evaluations:
                        raise RuntimeError(
                            "Adam search did not consume its exact budget"
                        )
                    search_results[label] = result
                    search_seconds[label] = time.perf_counter() - search_started

                batch_rows = []
                batch_rows.extend(
                    _evaluate_coordinates(
                        starts["fps"],
                        samples,
                        cloud_metadata,
                        decoder=decoder,
                        stability=stability,
                        config=config,
                        constellation_size=constellation_size,
                        coordinate_bits=coordinate_bits,
                        dataset_name=dataset_name,
                        decoder_seed=decoder_seed,
                        split=split,
                        method_fields=[
                            {
                                "method": "fps",
                                "refiner_seed": None,
                                "representation_class": "strict-subset",
                                "decoder_evaluations": 0,
                                "selected_start": None,
                                "selection_source_chamfer_mse": None,
                            }
                            for _ in samples
                        ],
                        encode_seconds=[selector_seconds["fps"] / len(samples)]
                        * len(samples),
                        scratch_root=scratch_root,
                        metric_cache=metric_cache,
                        mode="fps",
                    )
                )
                primary = search_results["fps"]
                primary_seconds = (
                    selector_seconds["fps"] + search_seconds["fps"]
                ) / len(samples)
                batch_rows.extend(
                    _evaluate_coordinates(
                        primary.coordinates,
                        samples,
                        cloud_metadata,
                        decoder=decoder,
                        stability=stability,
                        config=config,
                        constellation_size=constellation_size,
                        coordinate_bits=coordinate_bits,
                        dataset_name=dataset_name,
                        decoder_seed=decoder_seed,
                        split=split,
                        method_fields=[
                            {
                                "method": "adam_ste",
                                "refiner_seed": None,
                                "representation_class": "free-coordinate",
                                "decoder_evaluations": config.adam_evaluations,
                                "selected_start": "fps",
                                "selection_source_chamfer_mse": float(
                                    primary.losses[index].item()
                                ),
                            }
                            for index in range(len(samples))
                        ],
                        encode_seconds=[primary_seconds] * len(samples),
                        scratch_root=scratch_root,
                        metric_cache=metric_cache,
                    )
                )
                selected = []
                for cloud_index in range(len(samples)):
                    candidates = [
                        {
                            "coordinates": search_results[label].coordinates[
                                cloud_index
                            ],
                            "source_chamfer_mse": float(
                                search_results[label].losses[cloud_index].item()
                            ),
                            "start": label,
                        }
                        for label in config.start_labels
                    ]
                    selected.append(_select_best_by_source_score(candidates))
                multistart_seconds = sum(
                    selector_seconds[label] + search_seconds[label]
                    for label in config.start_labels
                ) / len(samples)
                batch_rows.extend(
                    _evaluate_coordinates(
                        torch.stack(
                            [candidate["coordinates"] for candidate in selected]
                        ),
                        samples,
                        cloud_metadata,
                        decoder=decoder,
                        stability=stability,
                        config=config,
                        constellation_size=constellation_size,
                        coordinate_bits=coordinate_bits,
                        dataset_name=dataset_name,
                        decoder_seed=decoder_seed,
                        split=split,
                        method_fields=[
                            {
                                "method": "adam_multistart",
                                "refiner_seed": None,
                                "representation_class": "free-coordinate",
                                "decoder_evaluations": config.adam_evaluations
                                * len(config.start_labels),
                                "selected_start": candidate["start"],
                                "selection_source_chamfer_mse": candidate[
                                    "source_chamfer_mse"
                                ],
                            }
                            for candidate in selected
                        ],
                        encode_seconds=[multistart_seconds] * len(samples),
                        scratch_root=scratch_root,
                        metric_cache=metric_cache,
                    )
                )
                for refiner_seed, refiner in refiners.items():
                    _synchronize(device)
                    refiner_started = time.perf_counter()
                    coordinates = refiner(
                        source,
                        constellation_size,
                        decoder=decoder,
                        target=source,
                        num_output_points=stability.num_points,
                    )
                    _synchronize(device)
                    refiner_seconds = (time.perf_counter() - refiner_started) / len(
                        samples
                    )
                    batch_rows.extend(
                        _evaluate_coordinates(
                            coordinates,
                            samples,
                            cloud_metadata,
                            decoder=decoder,
                            stability=stability,
                            config=config,
                            constellation_size=constellation_size,
                            coordinate_bits=coordinate_bits,
                            dataset_name=dataset_name,
                            decoder_seed=decoder_seed,
                            split=split,
                            method_fields=[
                                {
                                    "method": "refiner",
                                    "refiner_seed": refiner_seed,
                                    "representation_class": "free-coordinate",
                                    "decoder_evaluations": None,
                                    "selected_start": None,
                                    "selection_source_chamfer_mse": None,
                                }
                                for _ in samples
                            ],
                            encode_seconds=[refiner_seconds] * len(samples),
                            scratch_root=scratch_root,
                            metric_cache=metric_cache,
                        )
                    )
                for row in batch_rows:
                    key = _row_key(row)
                    if key in completed:
                        continue
                    _append_row(rows_path, row)
                    rows.append(row)
                    completed.add(key)
        if _state_hash(decoder) != decoder_hash:
            raise RuntimeError("frozen stabilized decoder changed during rate sweep")

    expected_rows = _cell_expected_row_count(
        config, stability, datasets, constellation_size
    )
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"cell {_cell_id(constellation_size, coordinate_bits)} has "
            f"{len(rows)} rows; expected {expected_rows}"
        )
    summary = summarize_cell_rows(
        rows,
        config,
        constellation_size=constellation_size,
        coordinate_bits=coordinate_bits,
        gpcc_rows=gpcc_rows,
    )
    result = {
        "experiment": "025_rate_sweep_adam",
        **summary,
        "resumed_rows": resumed_rows,
        "per_cloud_rows": len(rows),
        "expected_per_cloud_rows": expected_rows,
        "model_records": model_records,
        "contract_checks": {
            "complete_cell": len(rows) == expected_rows,
            "no_duplicate_rows": len(rows) == len({_row_key(row) for row in rows}),
            "decoder_hashes_unchanged": True,
            "actual_stream_bytes_match_expected": all(
                row["stream_bytes"] == rate["stream_bytes"] for row in rows
            ),
            "header_payload_split_exact": all(
                row["header_bytes"] + row["payload_bytes"] == row["stream_bytes"]
                for row in rows
            ),
            "exact_stream_round_trip": all(
                row["serialized_round_trip_exact"] for row in rows
            ),
            "exact_coordinate_lattice": all(
                row["coordinates_on_exact_lattice"] for row in rows
            ),
            "source_only_optimization": all(
                row["source_only_optimization"] for row in rows
            ),
            "official_d1_d2_present": all(
                row.get("d1_mse") is not None and row.get("d2_mse") is not None
                for row in rows
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "per_cloud_path": str(rows_path),
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError("Experiment 025 cell scientific contract failed")
    _write_json_atomic(cell_dir / "cell_metrics.json", result)
    return result


def _gpcc_frontier(
    gpcc_rows: Sequence[Mapping[str, Any]],
    split: str,
    *,
    cloud_keys: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in gpcc_rows:
        if row["split"] in _reference_split(split):
            groups.setdefault(str(row["rate_point"]), []).append(row)
    complete_groups = {
        rate_point: group
        for rate_point, group in groups.items()
        if {(row["family"], row["model_id"], row["sample_id"]) for row in group}
        == cloud_keys
    }
    return [
        {
            "rate_point": rate_point,
            "payload_bytes": float(np.mean([row["payload_bytes"] for row in group])),
            "stream_bytes": float(np.mean([row["stream_bytes"] for row in group])),
            "d1_mse": float(np.mean([row["d1_mse"] for row in group])),
            "d2_mse": float(np.mean([row["d2_mse"] for row in group])),
        }
        for rate_point, group in sorted(complete_groups.items())
    ]


def _gate_a1(
    table: Sequence[Mapping[str, Any]],
    gpcc_rows: Sequence[Mapping[str, Any]],
    learned_rows: Sequence[Mapping[str, Any]],
    *,
    grid_complete: bool,
) -> dict[str, Any]:
    cloud_keys = {
        (row["family"], row["model_id"], row["sample_id"])
        for row in learned_rows
        if row["split"] == "validation" and row["method"] == PRIMARY_METHOD
    }
    reference = _gpcc_frontier(
        gpcc_rows,
        "validation",
        cloud_keys=cloud_keys,
    )
    candidates = [
        row
        for row in table
        if row["split"] == "validation"
        and row["method"] == PRIMARY_METHOD
        and row["stream_bytes"] < 80
    ]
    points = []
    for candidate in candidates:
        dominators = [
            point
            for point in reference
            if point["payload_bytes"] <= candidate["payload_bytes"]
            and point["d1_mse"] <= candidate["d1_mse"]
            and point["d2_mse"] <= candidate["d2_mse"]
            and (
                point["payload_bytes"] < candidate["payload_bytes"]
                or point["d1_mse"] < candidate["d1_mse"]
                or point["d2_mse"] < candidate["d2_mse"]
            )
        ]
        points.append(
            {
                "rate_point": candidate["rate_point"],
                "constellation_size": candidate["constellation_size"],
                "coordinate_bits": candidate["coordinate_bits"],
                "stream_bytes": candidate["stream_bytes"],
                "payload_bytes": candidate["payload_bytes"],
                "not_dominated_by_measured_gpcc": not dominators,
                "dominating_gpcc_rate_points": [
                    point["rate_point"] for point in dominators
                ],
            }
        )
    nondominated = sum(point["not_dominated_by_measured_gpcc"] for point in points)
    evaluable = grid_complete and bool(reference)
    passes = nondominated >= 4 if evaluable else None
    if not grid_complete:
        decision = "pending_incomplete_grid"
    elif not reference:
        decision = "not_evaluable_missing_gpcc_reference"
    else:
        decision = "continue_track_a" if passes else "stop_track_a"
    return {
        "gate": "G-A1",
        "definition": (
            "at least four validation Adam-64 stabilized points with complete "
            "streams below 80 bytes are not jointly dominated in payload bytes, "
            "official D1 MSE, and official D2 MSE by a measured G-PCC point"
        ),
        "reference_available": bool(reference),
        "grid_complete": grid_complete,
        "evaluable": evaluable,
        "eligible_points": len(points),
        "nondominated_points": nondominated,
        "required_nondominated_points": 4,
        "passes": passes,
        "decision": decision,
        "points": points,
    }


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Split | K | q | Method | Stream B | Payload B | D1 RMSE | D2 RMSE |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item["split"],
            item["stream_bytes"],
            item["constellation_size"],
            item["coordinate_bits"],
            item["method"],
        ),
    ):
        lines.append(
            "| {split} | {constellation_size} | {coordinate_bits} | {method} | "
            "{stream_bytes} | {payload_bytes} | {d1_rmse:.6g} | {d2_rmse:.6g} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def aggregate_rate_sweep(
    config: RateSweepExperimentConfig,
    *,
    stability: StabilityExperimentConfig | None = None,
    datasets: Mapping[str, Any] | None = None,
    resumed_rows: int = 0,
) -> dict[str, Any]:
    """Aggregate completed cell artifacts without running model inference."""

    stability = stability or StabilityExperimentConfig.from_json(
        Path(config.stability_config)
    )
    datasets = datasets or _datasets(stability)
    output_dir = Path(config.output_dir)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config") != _json_ready(asdict(config)):
            raise RuntimeError("rate-sweep aggregate config differs from run manifest")
    cell_results = []
    all_rows = []
    for constellation_size, coordinate_bits in _cells(config):
        cell_dir = output_dir / "cells" / _cell_id(constellation_size, coordinate_bits)
        rows_path = cell_dir / "rate_sweep_per_cloud.jsonl"
        if not rows_path.is_file():
            continue
        rows = _load_rows(rows_path)
        expected = _cell_expected_row_count(
            config, stability, datasets, constellation_size
        )
        if len(rows) != expected:
            continue
        metrics_path = cell_dir / "cell_metrics.json"
        if not metrics_path.is_file():
            continue
        cell_results.append(json.loads(metrics_path.read_text()))
        all_rows.extend(rows)
    table = [row for cell in cell_results for row in cell["table"]]
    comparisons = [row for cell in cell_results for row in cell.get("comparisons", [])]
    gpcc_rows = _load_gpcc_rows(
        Path(config.gpcc_reference_path) if config.gpcc_reference_path else None,
        input_points=stability.num_points,
    )
    expected_rows = sum(
        _cell_expected_row_count(config, stability, datasets, constellation_size)
        * len(config.coordinate_bits)
        for constellation_size in config.constellation_sizes
    )
    complete_grid = (
        len(cell_results) == len(_cells(config)) and len(all_rows) == expected_rows
    )
    gate = _gate_a1(
        table,
        gpcc_rows,
        all_rows,
        grid_complete=complete_grid,
    )
    result = {
        "experiment": "025_rate_sweep_adam",
        "dataset": str(
            _data_protocol(stability, datasets).get("dataset", stability.dataset_kind)
        ),
        "config": _json_ready(asdict(config)),
        "resumed_rows": resumed_rows,
        "per_cloud_rows": len(all_rows),
        "expected_per_cloud_rows": expected_rows,
        "completed_cells": len(cell_results),
        "expected_cells": len(_cells(config)),
        "cell_table": table,
        "comparisons": comparisons,
        "gate_g_a1": gate,
        "contract_checks": {
            "completed_cells_have_no_duplicate_rows": len(all_rows)
            == len(
                {
                    (
                        row["constellation_size"],
                        row["coordinate_bits"],
                        *_row_key(row),
                    )
                    for row in all_rows
                }
            ),
            "all_completed_rows_have_exact_rates": all(
                row["stream_bytes"]
                == expected_stream_bytes(
                    row["constellation_size"], row["coordinate_bits"]
                )
                and row["payload_bytes"]
                == expected_payload_bytes(
                    row["constellation_size"], row["coordinate_bits"]
                )
                for row in all_rows
            ),
            "complete_grid": complete_grid,
        },
        "per_cell_metrics": [
            str(output_dir / "cells" / cell["cell_id"] / "cell_metrics.json")
            for cell in cell_results
        ],
    }
    curve = {
        "schema_version": 1,
        "experiment": result["experiment"],
        "dataset": result["dataset"],
        "rate_definition": "complete serialized stream bits / input source point",
        "payload_rate_definition": (
            "byte-aligned coordinate payload bits / input source point"
        ),
        "rows": table,
    }
    _write_json_atomic(output_dir / "rate_sweep_curve.json", curve)
    _write_json_atomic(output_dir / "rate_sweep_metrics.json", result)
    _write_text_atomic(output_dir / "rate_sweep_table.md", _markdown_table(table))
    return result


def run_rate_sweep_experiment(
    config: RateSweepExperimentConfig,
    *,
    device_name: str | None = None,
    cells: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Run or resume selected cells and rebuild the available curve summary."""

    requested_cells = tuple(cells) if cells is not None else _cells(config)
    unknown = set(requested_cells) - set(_cells(config))
    if unknown:
        raise ValueError(
            f"requested rate cells are outside the grid: {sorted(unknown)}"
        )
    if len(requested_cells) != len(set(requested_cells)):
        raise ValueError("requested rate cells must be unique")
    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    if set(config.decoder_seeds) - set(stability.decoder_seeds):
        raise ValueError("rate-sweep decoder seeds are absent from Experiment 019")
    if set(config.refiner_seeds) - set(stability.refiner_seeds):
        raise ValueError("rate-sweep refiner seeds are absent from Experiment 019")
    if max(config.constellation_sizes) > max(stability.training_constellation_sizes):
        raise ValueError("rate grid exceeds the variable decoder's trained capacity")
    if max(config.coordinate_bits) > stability.coordinate_bits:
        raise ValueError("rate grid cannot exceed the decoder training precision")
    if config.position_bits != stability.coordinate_bits:
        raise ValueError("official metric grid must match Experiment 019")
    executable = Path(config.pc_error_executable)
    if not executable.is_file() or not executable.stat().st_mode & 0o111:
        raise FileNotFoundError(f"pc_error is missing or not executable: {executable}")
    artifact_dir = Path(config.stability_artifact_dir)
    stability_metrics_path = artifact_dir / "stability_metrics.json"
    stability_metrics = json.loads(stability_metrics_path.read_text())
    if stability_metrics["config"] != _json_ready(asdict(stability)):
        raise RuntimeError("Experiment 019 artifact config differs from checked config")
    if not all(stability_metrics["contract_checks"].values()):
        raise RuntimeError("Experiment 019 artifact has a failed scientific contract")
    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    if data_protocol != stability_metrics["data_protocol"]:
        raise RuntimeError("Experiment 019 data identity changed before rate sweep")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpcc_path = Path(config.gpcc_reference_path) if config.gpcc_reference_path else None
    gpcc_rows = _load_gpcc_rows(gpcc_path, input_points=stability.num_points)
    manifest = {
        "experiment": "025_rate_sweep_adam",
        "config": _json_ready(asdict(config)),
        "stability_config_sha256": file_sha256(stability_path),
        "stability_metrics_sha256": file_sha256(stability_metrics_path),
        "pc_error_sha256": file_sha256(executable),
        "gpcc_reference_sha256": file_sha256(gpcc_path) if gpcc_path else None,
        "normalized_gpcc_reference_sha256": _gpcc_reference_digest(gpcc_rows),
        "data_protocol": data_protocol,
        "decoder_reuse": {
            "kind": "shared_variable_cardinality_decoder",
            "training_constellation_sizes": list(
                stability.training_constellation_sizes
            ),
            "target_constellation_sizes": list(config.constellation_sizes),
            "interpolated_target_sizes": sorted(
                set(config.constellation_sizes)
                - set(stability.training_constellation_sizes)
            ),
            "k8_checkpoints_reused_without_retraining": (
                stability.constellation_size == 8
                and config.refiner_constellation_size == 8
            ),
        },
    }
    _write_manifest_once(output_dir / "run_manifest.json", manifest)
    device = select_device(device_name)
    dataset_name = str(data_protocol.get("dataset", stability.dataset_kind))
    resumed_rows = 0
    for constellation_size, coordinate_bits in requested_cells:
        cell = _run_cell(
            config,
            stability,
            datasets,
            constellation_size=constellation_size,
            coordinate_bits=coordinate_bits,
            dataset_name=dataset_name,
            device=device,
            gpcc_rows=gpcc_rows,
        )
        resumed_rows += int(cell["resumed_rows"])
    return aggregate_rate_sweep(
        config,
        stability=stability,
        datasets=datasets,
        resumed_rows=resumed_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_025_rate_sweep_smoke.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--cell", nargs=2, type=int, metavar=("K", "Q"))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--stability-artifact-dir", type=Path)
    parser.add_argument("--pc-error", type=Path)
    parser.add_argument("--gpcc-reference", type=Path)
    parser.add_argument("--max-clouds-per-split", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = RateSweepExperimentConfig.from_json(args.config)
    if args.stability_artifact_dir is not None:
        config = replace(
            config, stability_artifact_dir=str(args.stability_artifact_dir)
        )
    if args.pc_error is not None:
        config = replace(config, pc_error_executable=str(args.pc_error))
    if args.gpcc_reference is not None:
        config = replace(config, gpcc_reference_path=str(args.gpcc_reference))
    if args.max_clouds_per_split is not None:
        config = replace(config, max_clouds_per_split=args.max_clouds_per_split)
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    if args.aggregate_only:
        result = aggregate_rate_sweep(config)
    else:
        cells = [tuple(args.cell)] if args.cell is not None else None
        result = run_rate_sweep_experiment(
            config,
            device_name=args.device,
            cells=cells,
        )
    print(
        json.dumps(
            {
                "completed_cells": result["completed_cells"],
                "expected_cells": result["expected_cells"],
                "rows": result["per_cloud_rows"],
                "gate_g_a1": result["gate_g_a1"],
                "metrics": str(Path(config.output_dir) / "rate_sweep_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
