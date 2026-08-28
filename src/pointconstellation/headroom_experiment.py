"""Experiment 022: multi-start Adam/STE inference headroom and timing Pareto."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from pointconstellation.bitstream import (
    decode_constellation,
    encode_constellation,
    expected_stream_bytes,
)
from pointconstellation.codecs import run_pc_error
from pointconstellation.data import file_sha256
from pointconstellation.models.gradient_free import SearchResult, adam_ste_search
from pointconstellation.official_stability import (
    OfficialStabilityConfig,
    _bootstrap_comparison,
    _load_models,
    _synchronize,
)
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.selection_baselines import SELECTION_METHODS
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _per_cloud_chamfer,
    stability_config_matches_artifact,
)
from pointconstellation.train import select_device

DEFAULT_BUDGETS = (16, 64, 256)
DEFAULT_START_METHODS = ("fps", "kmeans")
DEFAULT_RANDOM_START_SEEDS = (101, 211)
SUPPORTED_SPLITS = ("validation", "ood")
SUPPORTED_TIMING_DEVICES = ("mps", "cpu", "cuda")
SUMMARY_METRICS = (
    "source_chamfer_mse",
    "fresh_chamfer_mse",
    "d1_mse",
    "d2_mse",
)


@dataclass(frozen=True)
class HeadroomExperimentConfig:
    """Validated fixed-rate per-cloud optimization configuration."""

    stability_config: str = "configs/experiment_019_stability_modelnet40.json"
    stability_artifact_dir: str = "artifacts/local/experiment_019_stability_modelnet40"
    pc_error_executable: str = "artifacts/tools/mpeg-pcc-dmetric/build/Release/pc_error"
    position_bits: int = 12
    timeout_seconds: float = 120.0
    decoder_seeds: tuple[int, ...] = (7, 17, 29, 41, 53, 67)
    refiner_seeds: tuple[int, ...] = (101, 211, 307)
    start_methods: tuple[str, ...] = DEFAULT_START_METHODS
    random_start_seeds: tuple[int, ...] = DEFAULT_RANDOM_START_SEEDS
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    adam_learning_rate: float = 0.03
    selection_seed: int = 20_260_822
    splits: tuple[str, ...] = SUPPORTED_SPLITS
    max_clouds_per_split: int | None = None
    batch_size: int = 4
    timing_devices: tuple[str, ...] = ("cpu",)
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_822
    confidence_level: float = 0.95
    output_dir: str = "artifacts/local/experiment_022_headroom_modelnet40"

    def __post_init__(self) -> None:
        if not 2 <= self.position_bits <= 24:
            raise ValueError("position_bits must be between 2 and 24")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if len(self.decoder_seeds) < 2 or len(set(self.decoder_seeds)) != len(
            self.decoder_seeds
        ):
            raise ValueError("decoder_seeds must contain at least two unique seeds")
        if len(self.refiner_seeds) < 2 or len(set(self.refiner_seeds)) != len(
            self.refiner_seeds
        ):
            raise ValueError("refiner_seeds must contain at least two unique seeds")
        if (
            not self.start_methods
            or len(set(self.start_methods)) != len(self.start_methods)
            or set(self.start_methods) != set(DEFAULT_START_METHODS)
        ):
            raise ValueError("start_methods must contain fps and kmeans exactly once")
        if (
            len(self.random_start_seeds) != 2
            or len(set(self.random_start_seeds)) != 2
            or min(self.random_start_seeds) < 0
        ):
            raise ValueError(
                "random_start_seeds must contain two unique nonnegative seeds"
            )
        if (
            not self.budgets
            or len(set(self.budgets)) != len(self.budgets)
            or min(self.budgets) < 1
            or tuple(sorted(self.budgets)) != self.budgets
        ):
            raise ValueError("budgets must be unique, positive, and increasing")
        if self.adam_learning_rate <= 0:
            raise ValueError("adam_learning_rate must be positive")
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
        if (
            not self.timing_devices
            or len(set(self.timing_devices)) != len(self.timing_devices)
            or set(self.timing_devices) - set(SUPPORTED_TIMING_DEVICES)
        ):
            raise ValueError(
                "timing_devices must contain unique cpu, mps, and/or cuda entries"
            )
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

    @classmethod
    def from_json(cls, path: Path) -> HeadroomExperimentConfig:
        values = json.loads(path.read_text())
        for key in (
            "decoder_seeds",
            "refiner_seeds",
            "start_methods",
            "random_start_seeds",
            "budgets",
            "splits",
            "timing_devices",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @property
    def start_labels(self) -> tuple[str, ...]:
        return (
            *self.start_methods,
            *(f"random_seed_{s}" for s in self.random_start_seeds),
        )


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _official_config(config: HeadroomExperimentConfig) -> OfficialStabilityConfig:
    return OfficialStabilityConfig(
        stability_config=config.stability_config,
        stability_artifact_dir=config.stability_artifact_dir,
        pc_error_executable=config.pc_error_executable,
        position_bits=config.position_bits,
        timeout_seconds=config.timeout_seconds,
        decoder_seeds=config.decoder_seeds,
        refiner_seeds=config.refiner_seeds,
        splits=config.splits,
        max_clouds_per_split=config.max_clouds_per_split,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
        output_dir=config.output_dir,
    )


def _metadata(sample: Mapping[str, Any], fallback: int) -> dict[str, Any]:
    def value(name: str, default: Any) -> Any:
        item = sample.get(name, default)
        if isinstance(item, Tensor):
            if item.numel() != 1:
                raise ValueError(f"sample metadata {name} must be scalar")
            return item.item()
        return item

    sample_id = int(value("sample_id", fallback))
    return {
        "family": str(value("family", value("category", "unknown"))),
        "model_id": str(value("model_id", sample_id)),
        "sample_id": sample_id,
    }


def _sample_tensor(sample: Mapping[str, Any], *names: str) -> Tensor | None:
    for name in names:
        value = sample.get(name)
        if isinstance(value, Tensor):
            return value
    return None


def _batch_tensors(
    samples: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sources = []
    fresh = []
    normals = []
    for sample in samples:
        source = _sample_tensor(sample, "source_points", "points")
        if source is None:
            raise ValueError("sample is missing source points")
        fresh_points = _sample_tensor(sample, "fresh_points")
        source_normals = _sample_tensor(sample, "source_normals", "normals")
        if source_normals is None:
            raise ValueError("sample is missing source normals")
        sources.append(source)
        fresh.append(source if fresh_points is None else fresh_points)
        normals.append(source_normals)
    return (
        torch.stack(sources).to(device),
        torch.stack(fresh).to(device),
        torch.stack(normals).to(device),
    )


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["split"],
        row["method"],
        row["decoder_seed"],
        row.get("refiner_seed"),
        row.get("start"),
        row.get("budget"),
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
        raise RuntimeError("headroom JSONL contains duplicate rows")
    return rows


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _start_seed(
    config: HeadroomExperimentConfig,
    *,
    start: str,
    split: str,
    metadata: Mapping[str, Any],
) -> int:
    identity = json.dumps(
        [
            config.selection_seed,
            start,
            split,
            metadata["family"],
            metadata["model_id"],
            metadata["sample_id"],
        ],
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (
        2**63
    )


def _initial_starts(
    source: Tensor,
    metadata: Sequence[Mapping[str, Any]],
    config: HeadroomExperimentConfig,
    *,
    split: str,
    constellation_size: int,
    bits: int,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    starts: dict[str, Tensor] = {}
    elapsed: dict[str, float] = {}
    for label in config.start_labels:
        method = "random_best_of_1" if label.startswith("random_seed_") else label
        _synchronize(source.device)
        started = time.perf_counter()
        candidates = []
        for cloud, cloud_metadata in zip(source, metadata, strict=True):
            seed = _start_seed(
                config,
                start=label,
                split=split,
                metadata=cloud_metadata,
            )
            candidates.append(
                SELECTION_METHODS[method](
                    cloud,
                    constellation_size,
                    bits,
                    seed,
                    None,
                )
            )
        _synchronize(source.device)
        starts[label] = torch.stack(candidates)
        elapsed[label] = time.perf_counter() - started
    return starts, elapsed


def _source_scorer(
    decoder: nn.Module,
    source: Tensor,
    *,
    num_output_points: int,
    chunk_size: int,
) -> Callable[[Tensor], Tensor]:
    """Return the source-only symmetric squared-Chamfer search objective."""

    def score(coordinates: Tensor) -> Tensor:
        reconstruction = decoder(
            coordinates,
            num_output_points=num_output_points,
        )
        return _per_cloud_chamfer(
            reconstruction,
            source,
            chunk_size=chunk_size,
        )

    return score


def _search_adam_start(
    score: Callable[[Tensor], Tensor],
    initial: Tensor,
    *,
    bits: int,
    budget: int,
    learning_rate: float,
) -> SearchResult:
    """Run one start under a budget that counts every decoder scorer call."""

    return adam_ste_search(
        score,
        initial,
        bits=bits,
        decoder_evaluation_budget=budget,
        learning_rate=learning_rate,
    )


def _search_batch_on_device(
    decoder: nn.Module,
    samples: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    stability: StabilityExperimentConfig,
    config: HeadroomExperimentConfig,
    *,
    split: str,
    device: torch.device,
) -> tuple[
    Tensor,
    dict[int, dict[str, SearchResult]],
    dict[int, dict[str, float]],
    dict[str, float],
]:
    source, _, _ = _batch_tensors(samples, device)
    starts, selector_seconds = _initial_starts(
        source,
        metadata,
        config,
        split=split,
        constellation_size=stability.constellation_size,
        bits=stability.coordinate_bits,
    )
    score = _source_scorer(
        decoder,
        source,
        num_output_points=stability.num_points,
        chunk_size=stability.distance_chunk_size,
    )
    results: dict[int, dict[str, SearchResult]] = {}
    timings: dict[int, dict[str, float]] = {}
    for budget in config.budgets:
        results[budget] = {}
        timings[budget] = {}
        for label in config.start_labels:
            _synchronize(device)
            started = time.perf_counter()
            result = _search_adam_start(
                score,
                starts[label],
                bits=stability.coordinate_bits,
                budget=budget,
                learning_rate=config.adam_learning_rate,
            )
            _synchronize(device)
            search_seconds = time.perf_counter() - started
            if result.decoder_evaluations_per_cloud > budget:
                raise RuntimeError("Adam search exceeded its decoder evaluation budget")
            results[budget][label] = result
            timings[budget][label] = (selector_seconds[label] + search_seconds) / len(
                source
            )
    return (
        starts["fps"],
        results,
        timings,
        {label: seconds / len(source) for label, seconds in selector_seconds.items()},
    )


def _select_best_by_source_score(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select one candidate while reading only its source-score field."""

    if not candidates:
        raise ValueError("multi-start selection requires at least one candidate")
    return min(candidates, key=lambda candidate: float(candidate["source_chamfer_mse"]))


def _serialize_batch(
    coordinates: Tensor,
    *,
    stability: StabilityExperimentConfig,
    mode: str,
) -> tuple[Tensor, list[bytes], bool, bool, float]:
    started = time.perf_counter()
    decoded = []
    streams = []
    exact = True
    lattice_exact = True
    for coordinate_row in coordinates.detach().cpu().numpy():
        stream = encode_constellation(
            coordinate_row,
            bits=stability.coordinate_bits,
            mode=mode,
            output_points=stability.num_points,
        )
        packet = decode_constellation(stream)
        levels = (1 << packet.bits) - 1
        lattice = (packet.coordinates + 1.0) * 0.5 * levels
        lattice_exact = lattice_exact and bool(
            np.all(np.abs(lattice - np.rint(lattice)) <= 1e-9)
        )
        exact = (
            exact
            and encode_constellation(
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
        exact,
        lattice_exact,
        time.perf_counter() - started,
    )


def _official_cache_key(
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
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = _official_cache_key(
            decoder_seed=int(row["decoder_seed"]),
            split=str(row["split"]),
            metadata=row,
            stream_sha256=str(row["stream_sha256"]),
        )
        values = {
            name: row[name]
            for name in row
            if name.startswith("d1_") or name.startswith("d2_")
        }
        values["official_metric_seconds"] = row["official_metric_seconds"]
        if key in cache and cache[key] != values:
            raise RuntimeError("identical reconstructed streams have different metrics")
        cache[key] = values
    return cache


def _evaluate_coordinates(
    coordinates: Tensor,
    samples: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    *,
    decoder: nn.Module,
    stability: StabilityExperimentConfig,
    config: HeadroomExperimentConfig,
    decoder_seed: int,
    split: str,
    method_fields: Sequence[Mapping[str, Any]],
    encode_seconds: Mapping[str, Sequence[float]],
    scratch_root: Path,
    metric_cache: dict[tuple[Any, ...], dict[str, Any]],
    mode: str = "free",
) -> list[dict[str, Any]]:
    if len(coordinates) != len(samples) or len(method_fields) != len(samples):
        raise ValueError("coordinate, sample, and method batches must align")
    source, fresh, normals = _batch_tensors(samples, coordinates.device)
    decoded, streams, exact, lattice_exact, serialization_seconds = _serialize_batch(
        coordinates,
        stability=stability,
        mode=mode,
    )
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
    rows = []
    serialization_per_cloud = serialization_seconds / len(samples)
    for index, (_, cloud_metadata, fields, stream) in enumerate(
        zip(samples, metadata, method_fields, streams, strict=True)
    ):
        stream_sha256 = hashlib.sha256(stream).hexdigest()
        cache_key = _official_cache_key(
            decoder_seed=decoder_seed,
            split=split,
            metadata=cloud_metadata,
            stream_sha256=stream_sha256,
        )
        if cache_key not in metric_cache:
            with tempfile.TemporaryDirectory(
                prefix=f"{split}-d{decoder_seed}-",
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
            metric_cache[cache_key] = {
                **official.metrics,
                "official_metric_seconds": official.elapsed_seconds,
            }
        timing = {
            f"encode_seconds_{device}": float(values[index]) + serialization_per_cloud
            for device, values in encode_seconds.items()
        }
        for device in SUPPORTED_TIMING_DEVICES:
            timing.setdefault(f"encode_seconds_{device}", None)
        rows.append(
            {
                "experiment": "022_headroom_bound",
                "quality_device": coordinates.device.type,
                "timing_columns_are_independent_device_reexecutions": True,
                "split": split,
                "decoder_seed": decoder_seed,
                **cloud_metadata,
                **fields,
                "constellation_size": stability.constellation_size,
                "coordinate_bits": stability.coordinate_bits,
                "representation_class": fields.get(
                    "representation_class", "free-coordinate"
                ),
                "bitstream_mode": mode,
                "stream_hex": stream.hex(),
                "stream_sha256": stream_sha256,
                "stream_bytes": len(stream),
                "actual_stream_bpp": 8.0 * len(stream) / stability.num_points,
                "source_chamfer_mse": float(source_losses[index].item()),
                "fresh_chamfer_mse": float(fresh_losses[index].item()),
                "source_only_optimization": True,
                "multi_start_selection_field": (
                    "selection_source_chamfer_mse"
                    if fields["method"] == "adam_multistart"
                    else None
                ),
                "serialized_round_trip_exact": exact,
                "coordinates_on_exact_lattice": lattice_exact,
                **timing,
                **metric_cache[cache_key],
            }
        )
    return rows


def _arm_specs(
    config: HeadroomExperimentConfig,
) -> list[tuple[str, str | None, int | None, int | None]]:
    specs: list[tuple[str, str | None, int | None, int | None]] = [
        ("fps", None, None, None)
    ]
    specs.extend(
        ("refiner", None, None, refiner_seed) for refiner_seed in config.refiner_seeds
    )
    specs.extend(
        ("adam_ste", start, budget, None)
        for budget in config.budgets
        for start in config.start_labels
    )
    specs.extend(
        ("adam_multistart", "multi_start", budget, None) for budget in config.budgets
    )
    return specs


def _expected_keys(
    config: HeadroomExperimentConfig,
    *,
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
            start,
            budget,
            cloud["family"],
            cloud["model_id"],
            cloud["sample_id"],
        )
        for cloud in metadata
        for method, start, budget, refiner_seed in _arm_specs(config)
    }


def _adam_arm_name(method: str, start: str, budget: int) -> str:
    if method == "adam_multistart":
        return f"adam_multistart:budget_{budget}"
    return f"adam_ste:{start}:budget_{budget}"


def _comparison(
    baseline: np.ndarray,
    candidate: np.ndarray,
    categories: np.ndarray,
    *,
    baseline_arm: str,
    candidate_arm: str,
    config: HeadroomExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    result = _bootstrap_comparison(
        baseline,
        candidate,
        categories,
        samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
        seed=seed,
    )
    result["baseline_arm"] = baseline_arm
    result["candidate_arm"] = candidate_arm
    result["candidate_replicates"] = result.pop("refiner_count")
    result["aggregate_baseline_rmse"] = result.pop("aggregate_fps_rmse_grid_units")
    result["aggregate_candidate_rmse"] = result.pop("aggregate_refiner_rmse_grid_units")
    result["decoders_candidate_better"] = result.pop("decoders_better_than_fps")
    result["every_decoder_candidate_better"] = result.pop(
        "every_decoder_better_than_fps"
    )
    for decoder_seed, per_decoder in zip(
        config.decoder_seeds,
        result["per_decoder"],
        strict=True,
    ):
        per_decoder["decoder_seed"] = decoder_seed
        per_decoder.pop("decoder_index")
        per_decoder["baseline_rmse"] = per_decoder.pop("fps_rmse_grid_units")
        per_decoder["candidate_rmse"] = per_decoder.pop("refiner_rmse_grid_units")
    return result


def _pareto_points(
    rows: Sequence[Mapping[str, Any]],
    config: HeadroomExperimentConfig,
    *,
    split: str,
    timing_device: str,
) -> list[dict[str, Any]]:
    timing_field = f"encode_seconds_{timing_device}"
    arms: list[tuple[str, list[Mapping[str, Any]]]] = []
    arms.append(("fps", [row for row in rows if row["method"] == "fps"]))
    arms.append(("refiner", [row for row in rows if row["method"] == "refiner"]))
    for budget in config.budgets:
        for start in config.start_labels:
            name = _adam_arm_name("adam_ste", start, budget)
            selected = [
                row
                for row in rows
                if row["method"] == "adam_ste"
                and row["start"] == start
                and row["budget"] == budget
            ]
            arms.append((name, selected))
        name = _adam_arm_name("adam_multistart", "multi_start", budget)
        selected = [
            row
            for row in rows
            if row["method"] == "adam_multistart" and row["budget"] == budget
        ]
        arms.append((name, selected))
    points = []
    for arm, arm_rows in arms:
        arm_rows = [row for row in arm_rows if row["split"] == split]
        if not arm_rows or any(row[timing_field] is None for row in arm_rows):
            continue
        points.append(
            {
                "arm": arm,
                "split": split,
                "timing_device": timing_device,
                "mean_encode_seconds_per_cloud": float(
                    np.mean([row[timing_field] for row in arm_rows])
                ),
                "d1_rmse_grid_units": math.sqrt(
                    float(np.mean([row["d1_mse"] for row in arm_rows]))
                ),
                "rows": len(arm_rows),
            }
        )
    for point in points:
        point["on_pareto_front"] = not any(
            other["mean_encode_seconds_per_cloud"]
            <= point["mean_encode_seconds_per_cloud"]
            and other["d1_rmse_grid_units"] <= point["d1_rmse_grid_units"]
            and (
                other["mean_encode_seconds_per_cloud"]
                < point["mean_encode_seconds_per_cloud"]
                or other["d1_rmse_grid_units"] < point["d1_rmse_grid_units"]
            )
            for other in points
            if other is not point
        )
    return sorted(
        points,
        key=lambda point: (
            point["mean_encode_seconds_per_cloud"],
            point["d1_rmse_grid_units"],
        ),
    )


def summarize_headroom_rows(
    rows: list[dict[str, Any]],
    config: HeadroomExperimentConfig,
    *,
    quality_device: str,
) -> dict[str, Any]:
    """Build paired headroom comparisons and the encode-time/D1 Pareto table."""

    indexed = {_row_key(row): row for row in rows}
    comparisons = []
    headroom = []
    pareto = []
    for split_index, split in enumerate(config.splits):
        cloud_keys = sorted(
            {
                (row["family"], row["model_id"], row["sample_id"])
                for row in rows
                if row["split"] == split
            }
        )
        if not cloud_keys:
            raise RuntimeError(f"no Experiment 022 rows for split={split}")
        categories = np.asarray([key[0] for key in cloud_keys])
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            fps = np.asarray(
                [
                    [
                        indexed[(split, "fps", decoder_seed, None, None, None, *cloud)][
                            metric
                        ]
                        for cloud in cloud_keys
                    ]
                    for decoder_seed in config.decoder_seeds
                ],
                dtype=np.float64,
            )
            refiner = np.asarray(
                [
                    [
                        [
                            indexed[
                                (
                                    split,
                                    "refiner",
                                    decoder_seed,
                                    refiner_seed,
                                    None,
                                    None,
                                    *cloud,
                                )
                            ][metric]
                            for cloud in cloud_keys
                        ]
                        for refiner_seed in config.refiner_seeds
                    ]
                    for decoder_seed in config.decoder_seeds
                ],
                dtype=np.float64,
            )
            arms = [
                ("adam_ste", start, budget)
                for budget in config.budgets
                for start in config.start_labels
            ] + [
                ("adam_multistart", "multi_start", budget) for budget in config.budgets
            ]
            for arm_index, (method, start, budget) in enumerate(arms):
                adam = np.asarray(
                    [
                        [
                            indexed[
                                (
                                    split,
                                    method,
                                    decoder_seed,
                                    None,
                                    start,
                                    budget,
                                    *cloud,
                                )
                            ][metric]
                            for cloud in cloud_keys
                        ]
                        for decoder_seed in config.decoder_seeds
                    ],
                    dtype=np.float64,
                )
                arm_name = _adam_arm_name(method, start, budget)
                seed = (
                    config.bootstrap_seed
                    + split_index * 100_000
                    + metric_index * 10_000
                    + arm_index * 10
                )
                comparisons.append(
                    {
                        "split": split,
                        "metric": metric,
                        "comparison_role": "adam_vs_fps",
                        "adam_arm": arm_name,
                        **_comparison(
                            fps,
                            adam[:, None, :],
                            categories,
                            baseline_arm="fps",
                            candidate_arm=arm_name,
                            config=config,
                            seed=seed,
                        ),
                    }
                )
                comparisons.append(
                    {
                        "split": split,
                        "metric": metric,
                        "comparison_role": "refiner_vs_adam",
                        "adam_arm": arm_name,
                        **_comparison(
                            adam,
                            refiner,
                            categories,
                            baseline_arm=arm_name,
                            candidate_arm="refiner",
                            config=config,
                            seed=seed + 1,
                        ),
                    }
                )
            best_budget = config.budgets[-1]
            best_adam = np.asarray(
                [
                    [
                        indexed[
                            (
                                split,
                                "adam_multistart",
                                decoder_seed,
                                None,
                                "multi_start",
                                best_budget,
                                *cloud,
                            )
                        ][metric]
                        for cloud in cloud_keys
                    ]
                    for decoder_seed in config.decoder_seeds
                ],
                dtype=np.float64,
            )
            fps_rmse = math.sqrt(float(fps.mean()))
            refiner_rmse = math.sqrt(float(refiner.mean()))
            adam_rmse = math.sqrt(float(best_adam.mean()))
            denominator = fps_rmse - adam_rmse
            headroom.append(
                {
                    "split": split,
                    "metric": metric,
                    "best_adam_budget": best_budget,
                    "fps_rmse": fps_rmse,
                    "refiner_rmse": refiner_rmse,
                    "best_multistart_adam_rmse": adam_rmse,
                    "fraction_of_headroom_recovered": (
                        (fps_rmse - refiner_rmse) / denominator
                        if denominator > 0.0
                        else None
                    ),
                    "fraction_defined": denominator > 0.0,
                }
            )
        for timing_device in config.timing_devices:
            pareto.extend(
                _pareto_points(
                    rows,
                    config,
                    split=split,
                    timing_device=timing_device,
                )
            )
    primary_split = "validation" if "validation" in config.splits else config.splits[0]
    primary_pareto = [
        point
        for point in pareto
        if point["split"] == primary_split
        and point["timing_device"] == quality_device
        and point["arm"] == "refiner"
    ]
    return {
        "comparison_definition": (
            "paired hierarchical category/cloud bootstrap with paired decoder "
            "draws; the common refiner-seed factor is resampled only in "
            "refiner comparisons; relative improvement is on aggregate RMSE"
        ),
        "comparisons": comparisons,
        "headroom_recovery": headroom,
        "pareto_table": pareto,
        "primary_gate": {
            "split": primary_split,
            "metric": "d1_mse",
            "timing_device": quality_device,
            "criterion": (
                "aggregate refiner is nondominated in encode seconds versus "
                "official D1 RMSE among FPS and all Adam start/budget arms"
            ),
            "refiner_on_pareto_front": bool(
                primary_pareto and primary_pareto[0]["on_pareto_front"]
            ),
            "passes": bool(primary_pareto and primary_pareto[0]["on_pareto_front"]),
        },
    }


def _timed_refiner_batch(
    decoder: nn.Module,
    refiner: nn.Module,
    samples: Sequence[Mapping[str, Any]],
    stability: StabilityExperimentConfig,
    *,
    device: torch.device,
) -> tuple[Tensor, float]:
    source, _, _ = _batch_tensors(samples, device)
    _synchronize(device)
    started = time.perf_counter()
    coordinates = refiner(
        source,
        stability.constellation_size,
        decoder=decoder,
        target=source,
        num_output_points=stability.num_points,
    )
    _synchronize(device)
    return coordinates, (time.perf_counter() - started) / len(source)


def run_headroom_experiment(
    config: HeadroomExperimentConfig,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate or resume the full source-only inference headroom factorial."""

    quality_device = select_device(device_name or config.timing_devices[0])
    if quality_device.type not in SUPPORTED_TIMING_DEVICES:
        raise ValueError("Experiment 022 quality device must be cpu, mps, or cuda")
    if quality_device.type not in config.timing_devices:
        config = replace(
            config,
            timing_devices=(quality_device.type, *config.timing_devices),
        )
    timing_devices = {name: select_device(name) for name in config.timing_devices}
    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    if not config.decoder_seeds or any(
        seed not in stability.decoder_seeds for seed in config.decoder_seeds
    ):
        raise ValueError(
            "headroom decoder seeds must be a subset of the Experiment 019 seeds"
        )
    if tuple(config.refiner_seeds) != tuple(stability.refiner_seeds):
        raise ValueError("headroom refiner seeds must match Experiment 019 exactly")
    if config.position_bits != stability.coordinate_bits:
        raise ValueError("official metric grid must match constellation precision")
    executable = Path(config.pc_error_executable)
    if not executable.is_file() or not executable.stat().st_mode & 0o111:
        raise FileNotFoundError(f"pc_error is missing or not executable: {executable}")

    artifact_dir = Path(config.stability_artifact_dir)
    stability_metrics_path = artifact_dir / "stability_metrics.json"
    stability_metrics = json.loads(stability_metrics_path.read_text())
    if not stability_config_matches_artifact(stability_metrics["config"], stability):
        raise RuntimeError("Experiment 019 artifact config differs from checked config")
    if not all(stability_metrics["contract_checks"].values()):
        raise RuntimeError("Experiment 019 artifact has a failed scientific contract")
    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    if data_protocol != stability_metrics["data_protocol"]:
        raise RuntimeError("Experiment 019 data identity changed before headroom run")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = output_dir / "metric_scratch"
    scratch_root.mkdir(exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "experiment": "022_headroom_bound",
        "config": _json_ready(asdict(config)),
        "quality_device": quality_device.type,
        "stability_config_sha256": file_sha256(stability_path),
        "stability_metrics_sha256": file_sha256(stability_metrics_path),
        "pc_error_sha256": file_sha256(executable),
        "data_protocol": data_protocol,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("existing Experiment 022 run manifest differs")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rows_path = output_dir / "headroom_per_cloud.jsonl"
    rows = _load_rows(rows_path)
    resumed_rows = len(rows)
    completed = {_row_key(row) for row in rows}
    metric_cache = _metric_cache(rows)
    expected_bytes = expected_stream_bytes(
        stability.constellation_size,
        stability.coordinate_bits,
    )
    if any(row["stream_bytes"] != expected_bytes for row in rows):
        raise RuntimeError("resumed stream size differs from the fixed operating point")

    official = _official_config(config)
    model_records = []
    started = time.perf_counter()
    for decoder_seed in config.decoder_seeds:
        decoders: dict[str, nn.Module] = {}
        decoder_hashes: dict[str, str] = {}
        for name, device in timing_devices.items():
            decoder, _, metadata = _load_models(
                stability,
                official,
                decoder_seed=decoder_seed,
                refiner_seed=None,
                device=device,
            )
            decoders[name] = decoder
            decoder_hashes[name] = _state_hash(decoder)
            model_records.append(
                {
                    "decoder_seed": decoder_seed,
                    "refiner_seed": None,
                    "device": name,
                    **metadata,
                }
            )
        refiners: dict[str, dict[int, tuple[nn.Module, nn.Module]]] = {
            name: {} for name in timing_devices
        }
        for name, device in timing_devices.items():
            for refiner_seed in config.refiner_seeds:
                pair_decoder, refiner, metadata = _load_models(
                    stability,
                    official,
                    decoder_seed=decoder_seed,
                    refiner_seed=refiner_seed,
                    device=device,
                )
                assert refiner is not None
                refiners[name][refiner_seed] = (pair_decoder, refiner)
                model_records.append(
                    {
                        "decoder_seed": decoder_seed,
                        "refiner_seed": refiner_seed,
                        "device": name,
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
                    batch_start,
                    min(batch_start + config.batch_size, cloud_count),
                )
                samples = [dataset[index] for index in indices]
                cloud_metadata = [
                    _metadata(sample, index)
                    for sample, index in zip(samples, indices, strict=True)
                ]
                expected_keys = _expected_keys(
                    config,
                    decoder_seed=decoder_seed,
                    split=split,
                    metadata=cloud_metadata,
                )
                if expected_keys <= completed:
                    continue

                search_coordinates: Tensor | None = None
                search_results: dict[int, dict[str, SearchResult]] | None = None
                search_timings: dict[str, dict[int, dict[str, float]]] = {}
                selector_timings: dict[str, dict[str, float]] = {}
                for name, device in timing_devices.items():
                    fps_coordinates, device_results, device_timings, selectors = (
                        _search_batch_on_device(
                            decoders[name],
                            samples,
                            cloud_metadata,
                            stability,
                            config,
                            split=split,
                            device=device,
                        )
                    )
                    search_timings[name] = device_timings
                    selector_timings[name] = selectors
                    if name == quality_device.type:
                        search_coordinates = fps_coordinates
                        search_results = device_results
                assert search_coordinates is not None and search_results is not None

                fps_timing = {
                    name: [selector_timings[name]["fps"]] * len(samples)
                    for name in timing_devices
                }
                fps_fields = [
                    {
                        "method": "fps",
                        "refiner_seed": None,
                        "start": None,
                        "budget": None,
                        "decoder_evaluations": 0,
                        "selection_source_chamfer_mse": None,
                        "selected_start": None,
                        "representation_class": "strict-subset",
                    }
                    for _ in samples
                ]
                candidate_rows = _evaluate_coordinates(
                    search_coordinates,
                    samples,
                    cloud_metadata,
                    decoder=decoders[quality_device.type],
                    stability=stability,
                    config=config,
                    decoder_seed=decoder_seed,
                    split=split,
                    method_fields=fps_fields,
                    encode_seconds=fps_timing,
                    scratch_root=scratch_root,
                    metric_cache=metric_cache,
                    mode="fps",
                )

                for budget in config.budgets:
                    for label in config.start_labels:
                        result = search_results[budget][label]
                        fields = [
                            {
                                "method": "adam_ste",
                                "refiner_seed": None,
                                "start": label,
                                "budget": budget,
                                "decoder_evaluations": (
                                    result.decoder_evaluations_per_cloud
                                ),
                                "selection_source_chamfer_mse": float(
                                    result.losses[index].item()
                                ),
                                "selected_start": label,
                            }
                            for index in range(len(samples))
                        ]
                        timing = {
                            name: [search_timings[name][budget][label]] * len(samples)
                            for name in timing_devices
                        }
                        candidate_rows.extend(
                            _evaluate_coordinates(
                                result.coordinates,
                                samples,
                                cloud_metadata,
                                decoder=decoders[quality_device.type],
                                stability=stability,
                                config=config,
                                decoder_seed=decoder_seed,
                                split=split,
                                method_fields=fields,
                                encode_seconds=timing,
                                scratch_root=scratch_root,
                                metric_cache=metric_cache,
                            )
                        )

                    selected = []
                    for cloud_index in range(len(samples)):
                        candidates = [
                            {
                                "coordinates": search_results[budget][
                                    label
                                ].coordinates[cloud_index],
                                "source_chamfer_mse": float(
                                    search_results[budget][label]
                                    .losses[cloud_index]
                                    .item()
                                ),
                                "start": label,
                            }
                            for label in config.start_labels
                        ]
                        selected.append(_select_best_by_source_score(candidates))
                    selected_coordinates = torch.stack(
                        [candidate["coordinates"] for candidate in selected]
                    )
                    fields = [
                        {
                            "method": "adam_multistart",
                            "refiner_seed": None,
                            "start": "multi_start",
                            "budget": budget,
                            "decoder_evaluations": budget * len(config.start_labels),
                            "selection_source_chamfer_mse": candidate[
                                "source_chamfer_mse"
                            ],
                            "selected_start": candidate["start"],
                        }
                        for candidate in selected
                    ]
                    timing = {
                        name: [
                            sum(
                                search_timings[name][budget][label]
                                for label in config.start_labels
                            )
                        ]
                        * len(samples)
                        for name in timing_devices
                    }
                    candidate_rows.extend(
                        _evaluate_coordinates(
                            selected_coordinates,
                            samples,
                            cloud_metadata,
                            decoder=decoders[quality_device.type],
                            stability=stability,
                            config=config,
                            decoder_seed=decoder_seed,
                            split=split,
                            method_fields=fields,
                            encode_seconds=timing,
                            scratch_root=scratch_root,
                            metric_cache=metric_cache,
                        )
                    )

                for refiner_seed in config.refiner_seeds:
                    refiner_coordinates: Tensor | None = None
                    timing = {}
                    for name, device in timing_devices.items():
                        pair_decoder, refiner = refiners[name][refiner_seed]
                        coordinates, elapsed = _timed_refiner_batch(
                            pair_decoder,
                            refiner,
                            samples,
                            stability,
                            device=device,
                        )
                        timing[name] = [elapsed] * len(samples)
                        if name == quality_device.type:
                            refiner_coordinates = coordinates
                    assert refiner_coordinates is not None
                    fields = [
                        {
                            "method": "refiner",
                            "refiner_seed": refiner_seed,
                            "start": None,
                            "budget": None,
                            "decoder_evaluations": None,
                            "selection_source_chamfer_mse": None,
                            "selected_start": None,
                        }
                        for _ in samples
                    ]
                    candidate_rows.extend(
                        _evaluate_coordinates(
                            refiner_coordinates,
                            samples,
                            cloud_metadata,
                            decoder=decoders[quality_device.type],
                            stability=stability,
                            config=config,
                            decoder_seed=decoder_seed,
                            split=split,
                            method_fields=fields,
                            encode_seconds=timing,
                            scratch_root=scratch_root,
                            metric_cache=metric_cache,
                        )
                    )

                for row in candidate_rows:
                    key = _row_key(row)
                    if key in completed:
                        continue
                    if row["stream_bytes"] != expected_bytes:
                        raise RuntimeError("Experiment 022 stream size is inconsistent")
                    _append_row(rows_path, row)
                    rows.append(row)
                    completed.add(key)

        for name, decoder in decoders.items():
            if _state_hash(decoder) != decoder_hashes[name]:
                raise RuntimeError("frozen decoder changed during Adam search")
        for name, pairs in refiners.items():
            for pair_decoder, _ in pairs.values():
                if _state_hash(pair_decoder) != decoder_hashes[name]:
                    raise RuntimeError("frozen decoder changed during refiner timing")

    expected_rows = 0
    for split in config.splits:
        count = len(datasets[split])
        if config.max_clouds_per_split is not None:
            count = min(count, config.max_clouds_per_split)
        expected_rows += count * len(config.decoder_seeds) * len(_arm_specs(config))
    summary = summarize_headroom_rows(
        rows,
        config,
        quality_device=quality_device.type,
    )
    result = {
        "experiment": "022_headroom_bound",
        "config": _json_ready(asdict(config)),
        "quality_device": quality_device.type,
        "timing_devices": list(config.timing_devices),
        "resumed_rows": resumed_rows,
        "per_cloud_rows": len(rows),
        "expected_per_cloud_rows": expected_rows,
        "expected_stream_bytes": expected_bytes,
        "score_contract": (
            "frozen-decoder symmetric squared Chamfer against encoder-visible "
            "source points only"
        ),
        "budget_contract": {
            "decoder_forward_backward_evaluation_counts_as_one": True,
            "diagnostic_and_official_evaluations_excluded": True,
            "per_start_budgets": list(config.budgets),
            "multi_start_total_evaluations": {
                str(budget): budget * len(config.start_labels)
                for budget in config.budgets
            },
        },
        "contract_checks": {
            "complete_factorial": len(rows) == expected_rows,
            "decoder_hashes_unchanged": True,
            "actual_stream_present": bool(
                rows and all(len(bytes.fromhex(row["stream_hex"])) > 0 for row in rows)
            ),
            "identical_declared_stream_bytes": bool(
                rows and all(row["stream_bytes"] == expected_bytes for row in rows)
            ),
            "exact_stream_round_trip": bool(
                rows and all(row["serialized_round_trip_exact"] for row in rows)
            ),
            "exact_coordinate_lattice": bool(
                rows and all(row["coordinates_on_exact_lattice"] for row in rows)
            ),
            "source_only_optimization": bool(
                rows and all(row["source_only_optimization"] for row in rows)
            ),
            "multi_start_selected_by_source_only": bool(
                rows
                and all(
                    row["multi_start_selection_field"] == "selection_source_chamfer_mse"
                    for row in rows
                    if row["method"] == "adam_multistart"
                )
            ),
            "requested_timing_columns_present": bool(
                rows
                and all(
                    row[f"encode_seconds_{device}"] is not None
                    for row in rows
                    for device in config.timing_devices
                )
            ),
        },
        "tool_identity": {
            "pc_error_path": str(executable),
            "pc_error_sha256": file_sha256(executable),
            "position_bits": config.position_bits,
        },
        "data_protocol": data_protocol,
        "model_records": model_records,
        "statistics": summary,
        "elapsed_seconds": time.perf_counter() - started,
        "per_cloud_path": str(rows_path),
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError("Experiment 022 scientific contract failed")
    (output_dir / "headroom_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_022_headroom_smoke.json"),
    )
    parser.add_argument("--device", choices=SUPPORTED_TIMING_DEVICES)
    parser.add_argument("--timing-devices", nargs="+", choices=SUPPORTED_TIMING_DEVICES)
    parser.add_argument("--stability-artifact-dir", type=Path)
    parser.add_argument("--pc-error", type=Path)
    parser.add_argument("--max-clouds-per-split", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = HeadroomExperimentConfig.from_json(args.config)
    if args.timing_devices is not None:
        config = replace(config, timing_devices=tuple(args.timing_devices))
    if args.stability_artifact_dir is not None:
        config = replace(
            config,
            stability_artifact_dir=str(args.stability_artifact_dir),
        )
    if args.pc_error is not None:
        config = replace(config, pc_error_executable=str(args.pc_error))
    if args.max_clouds_per_split is not None:
        config = replace(
            config,
            max_clouds_per_split=args.max_clouds_per_split,
        )
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    result = run_headroom_experiment(config, device_name=args.device)
    print(
        json.dumps(
            {
                "rows": result["per_cloud_rows"],
                "quality_device": result["quality_device"],
                "gate_passes": result["statistics"]["primary_gate"]["passes"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "headroom_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
