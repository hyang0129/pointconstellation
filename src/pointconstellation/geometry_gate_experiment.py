"""Experiment 031: procedural surface-versus-sample geometry gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor
from torch.utils.data import DataLoader

from pointconstellation.bitstream import decode_constellation, encode_constellation
from pointconstellation.data import (
    FAMILIES,
    ProceduralSurfaceDataset,
    analytic_surface_distances,
)
from pointconstellation.metrics import point_set_error_metrics, point_to_plane_mse
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.refiner import CompetitiveConstellationRefiner
from pointconstellation.refiner_experiment import (
    RefinerExperimentConfig,
    _fps,
    _state_hash,
    run_refiner_experiment,
)
from pointconstellation.train import select_device

TRAINING_PROTOCOLS = ("exact_sample", "independent_resampling")
EVALUATION_SPLITS = ("validation", "parameter_ood")
METHODS = ("fps", "refiner")


@dataclass(frozen=True)
class GeometryGateExperimentConfig:
    """Validated procedural Gate B training and evaluation configuration."""

    num_points: int = 256
    constellation_size: int = 16
    coordinate_bits: int = 12
    train_samples: int = 140
    validation_samples: int = 35
    parameter_ood_samples: int = 35
    batch_size: int = 7
    model_seeds: tuple[int, ...] = (7, 17, 29)
    data_seed: int = 31_031
    decoder_epochs: int = 2
    refiner_epochs: int = 2
    decoder_learning_rate: float = 1e-3
    refiner_learning_rate: float = 1e-3
    feature_width: int = 32
    num_heads: int = 4
    num_layers: int = 1
    recurrent_steps: int = 2
    responsibility_temperature: float = 0.2
    maximum_update: float = 0.1
    use_decoder_gradient: bool = True
    distance_chunk_size: int = 256
    boundary_band: float = 0.1
    recall_tolerance: float = 0.1
    perturbation_bins: tuple[int, ...] = (1, 2)
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 20_260_831
    confidence_level: float = 0.95
    output_dir: str = "artifacts/local/experiment_031_geometry_gate"

    def __post_init__(self) -> None:
        if self.num_points < 8:
            raise ValueError("num_points must be at least 8")
        if not 2 <= self.constellation_size <= self.num_points:
            raise ValueError("constellation_size must be between 2 and num_points")
        if not 2 <= self.coordinate_bits <= 24:
            raise ValueError("coordinate_bits must be between 2 and 24")
        counts = (
            self.train_samples,
            self.validation_samples,
            self.parameter_ood_samples,
            self.batch_size,
            self.decoder_epochs,
            self.refiner_epochs,
            self.distance_chunk_size,
        )
        if min(counts) < 1:
            raise ValueError("sample, batch, epoch, and chunk counts must be positive")
        if any(
            count < len(FAMILIES) or count % len(FAMILIES)
            for count in (
                self.train_samples,
                self.validation_samples,
                self.parameter_ood_samples,
            )
        ):
            raise ValueError(
                "each sample count must be a positive multiple of families"
            )
        if len(self.model_seeds) < 2 or len(set(self.model_seeds)) != len(
            self.model_seeds
        ):
            raise ValueError("model_seeds must contain at least two unique seeds")
        if self.feature_width < 4 or self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if self.num_layers < 1 or self.recurrent_steps < 1:
            raise ValueError("model layer and recurrent step counts must be positive")
        if min(self.decoder_learning_rate, self.refiner_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if self.responsibility_temperature <= 0 or self.maximum_update <= 0:
            raise ValueError("refiner temperature and update must be positive")
        if not 0.0 < self.boundary_band < 0.5:
            raise ValueError("boundary_band must be in (0, 0.5)")
        if self.recall_tolerance <= 0:
            raise ValueError("recall_tolerance must be positive")
        if (
            not self.perturbation_bins
            or len(set(self.perturbation_bins)) != len(self.perturbation_bins)
            or min(self.perturbation_bins) < 1
            or max(self.perturbation_bins) >= (1 << self.coordinate_bits) - 1
        ):
            raise ValueError(
                "perturbation_bins must be unique positive in-lattice offsets"
            )
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

    @classmethod
    def from_json(cls, path: Path) -> GeometryGateExperimentConfig:
        values = json.loads(path.read_text())
        for key in ("model_seeds", "perturbation_bins"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def perturb_quantized_coordinates(
    coordinates: ArrayLike,
    *,
    bits: int,
    bins: int,
    seed: int,
) -> NDArray[np.float32]:
    """Move every coordinate by a deterministic signed number of lattice bins."""

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("coordinates must have shape (K, 3) with K > 0")
    if not 2 <= bits <= 24:
        raise ValueError("bits must be between 2 and 24")
    if bins < 1:
        raise ValueError("bins must be positive")
    levels = (1 << bits) - 1
    if bins >= levels:
        raise ValueError("bins must be smaller than the number of lattice intervals")
    lattice = (values + 1.0) * 0.5 * levels
    rounded = np.rint(lattice)
    # float32 coordinates carry ~1e-7 relative error, i.e. up to ~3e-4 lattice
    # units at 12 bits; any offset far below half a bin is an exact lattice point.
    if not np.all(np.abs(lattice - rounded) <= 1e-2):
        raise ValueError("coordinates must lie on the declared quantization lattice")
    integers = rounded.astype(np.int64)
    rng = np.random.default_rng(seed)
    directions = rng.choice((-1, 1), size=integers.shape)
    directions = np.where(integers < bins, 1, directions)
    directions = np.where(integers > levels - bins, -1, directions)
    perturbed = np.clip(integers + bins * directions, 0, levels)
    return (2.0 * perturbed / levels - 1.0).astype(np.float32)


def _role_hash(points: Tensor) -> str:
    return hashlib.sha256(points.numpy().tobytes()).hexdigest()


def _datasets(
    config: GeometryGateExperimentConfig, training_protocol: str
) -> dict[str, ProceduralSurfaceDataset]:
    sizes = {
        "train": config.train_samples,
        "validation": config.validation_samples,
        "parameter_ood": config.parameter_ood_samples,
    }
    return {
        split: ProceduralSurfaceDataset(
            size,
            num_points=config.num_points,
            seed=config.data_seed,
            split=split,
            training_target=training_protocol,
            boundary_band=config.boundary_band,
        )
        for split, size in sizes.items()
    }


def _data_protocol(config: GeometryGateExperimentConfig) -> dict[str, Any]:
    datasets = _datasets(config, "exact_sample")
    splits = {}
    all_independent = True
    for split, dataset in datasets.items():
        records = []
        for index in range(len(dataset)):
            sample = dataset[index]
            hashes = {
                "x_a_sha256": _role_hash(sample["source_points"]),
                "x_b_sha256": _role_hash(sample["independent_points"]),
                "x_c_sha256": _role_hash(sample["fresh_points"]),
            }
            all_independent = all_independent and len(set(hashes.values())) == 3
            records.append(
                {
                    "sample_id": index,
                    "family": sample["family"],
                    **hashes,
                }
            )
        digest = hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        splits[split] = {"count": len(records), "sha256": digest, "records": records}
    return {
        "data_seed": config.data_seed,
        "roles": {
            "x_a": "encoder input and legal decoder-gradient target",
            "x_b": "independent training/evaluation target",
            "x_c": "fresh evaluation-only sample",
        },
        "all_role_samples_byte_distinct": all_independent,
        "splits": splits,
    }


def _refiner_config(
    config: GeometryGateExperimentConfig,
    *,
    model_seed: int,
    output_dir: Path,
) -> RefinerExperimentConfig:
    return RefinerExperimentConfig(
        num_points=config.num_points,
        input_sizes=(config.num_points,),
        constellation_sizes=(config.constellation_size,),
        bits=config.coordinate_bits,
        train_samples=config.train_samples,
        validation_samples=config.validation_samples,
        parameter_ood_samples=config.parameter_ood_samples,
        batch_size=config.batch_size,
        decoder_epochs=config.decoder_epochs,
        refiner_epochs=config.refiner_epochs,
        decoder_learning_rate=config.decoder_learning_rate,
        refiner_learning_rate=config.refiner_learning_rate,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        recurrent_steps=config.recurrent_steps,
        responsibility_temperature=config.responsibility_temperature,
        maximum_update=config.maximum_update,
        use_decoder_gradient=config.use_decoder_gradient,
        decoder_gradient_chunk_size=config.distance_chunk_size,
        run_internal_evaluation=False,
        data_seed=config.data_seed,
        seed=model_seed,
        output_dir=str(output_dir),
    )


def _loader_factory(
    datasets: Mapping[str, ProceduralSurfaceDataset],
    *,
    batch_size: int,
    seed: int,
):
    def make() -> tuple[DataLoader, DataLoader, DataLoader]:
        generator = torch.Generator().manual_seed(seed)
        return (
            DataLoader(
                datasets["train"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=generator,
            ),
            DataLoader(
                datasets["validation"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            ),
            DataLoader(
                datasets["parameter_ood"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            ),
        )

    return make


def _models(
    config: GeometryGateExperimentConfig,
    artifact_dir: Path,
    device: torch.device,
) -> tuple[VariableConstellationDecoder, CompetitiveConstellationRefiner]:
    decoder = VariableConstellationDecoder(
        config.num_points,
        config.constellation_size,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    refiner = CompetitiveConstellationRefiner(
        config.constellation_size,
        bits=config.coordinate_bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        recurrent_steps=config.recurrent_steps,
        responsibility_temperature=config.responsibility_temperature,
        maximum_update=config.maximum_update,
        use_decoder_gradient=config.use_decoder_gradient,
        decoder_gradient_chunk_size=config.distance_chunk_size,
    ).to(device)
    decoder.load_state_dict(
        torch.load(artifact_dir / "decoder.pt", map_location=device, weights_only=True)[
            "model"
        ]
    )
    refiner.load_state_dict(
        torch.load(artifact_dir / "refiner.pt", map_location=device, weights_only=True)[
            "model"
        ]
    )
    decoder.eval().requires_grad_(False)
    refiner.eval().requires_grad_(False)
    return decoder, refiner


def _coverage_recall(
    reconstruction: NDArray[np.float64],
    target: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    tolerance: float,
    chunk_size: int,
) -> float | None:
    selected = target[mask]
    if not len(selected):
        return None
    recalled = 0
    squared_tolerance = tolerance**2
    for start in range(0, len(selected), chunk_size):
        chunk = selected[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - reconstruction[None, :, :]) ** 2, axis=2)
        recalled += int(np.count_nonzero(squared.min(axis=1) <= squared_tolerance))
    return recalled / len(selected)


def _metrics(
    reconstruction: NDArray[np.float32],
    coordinates: NDArray[np.float32],
    sample: Mapping[str, Any],
    dataset: ProceduralSurfaceDataset,
    *,
    config: GeometryGateExperimentConfig,
) -> dict[str, float | None]:
    targets = {
        "x_a": (sample["source_points"], sample["source_normals"]),
        "x_b": (sample["independent_points"], sample["independent_normals"]),
        "x_c": (sample["fresh_points"], sample["fresh_normals"]),
    }
    result: dict[str, float | None] = {}
    for role, (points, normals) in targets.items():
        point_values = points.numpy()
        values = point_set_error_metrics(
            reconstruction, point_values, chunk_size=config.distance_chunk_size
        )
        result.update({f"{role}_{key}": value for key, value in values.items()})
        result[f"{role}_d2_mse"] = point_to_plane_mse(
            reconstruction,
            point_values,
            normals.numpy(),
            chunk_size=config.distance_chunk_size,
        )
    surface = dataset.surface(int(sample["sample_id"]))
    distances = analytic_surface_distances(reconstruction, surface)
    constellation_distances = analytic_surface_distances(coordinates, surface)
    fresh = sample["fresh_points"].numpy().astype(np.float64)
    result.update(
        {
            "surface_mse": float(np.mean(distances**2)),
            "surface_rmse": float(np.sqrt(np.mean(distances**2))),
            "constellation_surface_rmse": float(
                np.sqrt(np.mean(constellation_distances**2))
            ),
            "boundary_recall": _coverage_recall(
                reconstruction.astype(np.float64),
                fresh,
                sample["fresh_boundary_mask"].numpy(),
                tolerance=config.recall_tolerance,
                chunk_size=config.distance_chunk_size,
            ),
            "thin_structure_recall": _coverage_recall(
                reconstruction.astype(np.float64),
                fresh,
                sample["fresh_thin_structure_mask"].numpy(),
                tolerance=config.recall_tolerance,
                chunk_size=config.distance_chunk_size,
            ),
        }
    )
    return result


def _perturbation_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _serialized(
    coordinates: NDArray[np.float32],
    *,
    config: GeometryGateExperimentConfig,
    mode: str,
) -> tuple[NDArray[np.float32], bytes, bool]:
    stream = encode_constellation(
        coordinates,
        bits=config.coordinate_bits,
        mode=mode,
        output_points=config.num_points,
    )
    packet = decode_constellation(stream)
    levels = (1 << packet.bits) - 1
    lattice = (packet.coordinates + 1.0) * 0.5 * levels
    exact = bool(np.all(np.abs(lattice - np.rint(lattice)) <= 1e-9))
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
    return packet.coordinates.astype(np.float32), stream, exact


def _evaluate_models(
    config: GeometryGateExperimentConfig,
    datasets: Mapping[str, ProceduralSurfaceDataset],
    *,
    training_protocol: str,
    model_seed: int,
    decoder: VariableConstellationDecoder,
    refiner: CompetitiveConstellationRefiner,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    decoder_hash = _state_hash(decoder)
    for split in EVALUATION_SPLITS:
        dataset = datasets[split]
        for sample_index in range(len(dataset)):
            sample = dataset[sample_index]
            source = sample["source_points"].unsqueeze(0).to(device)
            with torch.no_grad():
                method_coordinates = {
                    "fps": _fps(
                        source, config.constellation_size, config.coordinate_bits
                    ),
                    "refiner": refiner(
                        source,
                        config.constellation_size,
                        decoder=decoder,
                        target=source,
                        num_output_points=config.num_points,
                    ),
                }
            for method, tensor_coordinates in method_coordinates.items():
                base = tensor_coordinates[0].detach().cpu().numpy()
                for bins in (0, *config.perturbation_bins):
                    coordinates = base
                    if bins:
                        coordinates = perturb_quantized_coordinates(
                            base,
                            bits=config.coordinate_bits,
                            bins=bins,
                            seed=_perturbation_seed(
                                config.data_seed,
                                training_protocol,
                                model_seed,
                                split,
                                sample_index,
                                method,
                                bins,
                            ),
                        )
                    mode = "fps" if method == "fps" and bins == 0 else "free"
                    decoded, stream, exact = _serialized(
                        coordinates, config=config, mode=mode
                    )
                    with torch.no_grad():
                        reconstruction = decoder(
                            torch.from_numpy(decoded).unsqueeze(0).to(device),
                            num_output_points=config.num_points,
                        )[0]
                    reconstruction_values = (
                        reconstruction.detach().cpu().numpy().astype(np.float32)
                    )
                    rows.append(
                        {
                            "training_protocol": training_protocol,
                            "model_seed": model_seed,
                            "split": split,
                            "family": str(sample["family"]),
                            "sample_id": int(sample["sample_id"]),
                            "method": method,
                            "perturbation_bins": bins,
                            "representation_class": (
                                "strict-subset"
                                if method == "fps" and bins == 0
                                else (
                                    "free-coordinate"
                                    if bins == 0
                                    else "post-hoc-lattice-perturbation"
                                )
                            ),
                            "constellation_size": config.constellation_size,
                            "coordinate_bits": config.coordinate_bits,
                            "stream_bytes": len(stream),
                            "stream_hex": stream.hex(),
                            "stream_sha256": hashlib.sha256(stream).hexdigest(),
                            "actual_stream_bpp": 8.0 * len(stream) / config.num_points,
                            "serialized_round_trip_exact": exact,
                            "coordinates_on_exact_lattice": exact,
                            "encoder_visible_role": "x_a",
                            "outer_training_target_role": (
                                "x_a" if training_protocol == "exact_sample" else "x_b"
                            ),
                            "fresh_evaluation_role": "x_c",
                            **_metrics(
                                reconstruction_values,
                                decoded,
                                sample,
                                dataset,
                                config=config,
                            ),
                        }
                    )
    if _state_hash(decoder) != decoder_hash:
        raise RuntimeError("frozen decoder changed during Gate B evaluation")
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["training_protocol"],
            row["model_seed"],
            row["split"],
            row["method"],
            row["perturbation_bins"],
            row["family"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    fields = (
        "x_a_chamfer_mse",
        "x_b_chamfer_mse",
        "x_c_chamfer_mse",
        "x_a_d2_mse",
        "x_b_d2_mse",
        "x_c_d2_mse",
        "surface_mse",
        "constellation_surface_rmse",
        "x_c_p90_euclidean",
        "x_c_p99_euclidean",
        "x_c_hausdorff",
        "boundary_recall",
        "thin_structure_recall",
    )
    for key, group in sorted(groups.items()):
        values: dict[str, Any] = {}
        for field in fields:
            present = [row[field] for row in group if row[field] is not None]
            values[field] = float(np.mean(present)) if present else None
        for field in (
            "x_a_chamfer_mse",
            "x_b_chamfer_mse",
            "x_c_chamfer_mse",
            "x_a_d2_mse",
            "x_b_d2_mse",
            "x_c_d2_mse",
            "surface_mse",
        ):
            values[field.removesuffix("_mse") + "_rmse"] = math.sqrt(values[field])
        summaries.append(
            {
                "training_protocol": key[0],
                "model_seed": key[1],
                "split": key[2],
                "method": key[3],
                "perturbation_bins": key[4],
                "family": key[5],
                "clouds": len(group),
                **values,
            }
        )
    return summaries


def _cloud_draw(
    families: NDArray[np.str_], rng: np.random.Generator
) -> NDArray[np.int64]:
    unique = np.unique(families)
    drawn_families = unique[rng.integers(0, len(unique), size=len(unique))]
    indices = []
    for family in drawn_families:
        candidates = np.flatnonzero(families == family)
        indices.extend(
            candidates[rng.integers(0, len(candidates), size=len(candidates))].tolist()
        )
    return np.asarray(indices, dtype=np.int64)


def _paired_gate_comparison(
    baseline: NDArray[np.float64],
    candidate: NDArray[np.float64],
    families: NDArray[np.str_],
    *,
    config: GeometryGateExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    def improvement(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
        first_rmse = math.sqrt(float(first.mean()))
        second_rmse = math.sqrt(float(second.mean()))
        return 100.0 * (first_rmse - second_rmse) / max(first_rmse, 1e-12)

    per_seed = np.asarray(
        [
            improvement(baseline[index], candidate[index])
            for index in range(len(baseline))
        ]
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(config.bootstrap_samples, dtype=np.float64)
    for index in range(config.bootstrap_samples):
        seed_draw = rng.integers(0, len(baseline), size=len(baseline))
        cloud_draw = _cloud_draw(families, rng)
        draws[index] = improvement(
            baseline[seed_draw[:, None], cloud_draw[None, :]],
            candidate[seed_draw[:, None], cloud_draw[None, :]],
        )
    alpha = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(draws, (alpha, 1.0 - alpha))
    return {
        "relative_rmse_improvement_percent": improvement(baseline, candidate),
        "confidence_interval_lower_percent": float(lower),
        "confidence_interval_upper_percent": float(upper),
        "every_seed_better_than_fps": bool(np.all(per_seed > 0.0)),
        "per_seed_relative_rmse_improvement_percent": per_seed.tolist(),
        "passes": bool(np.all(per_seed > 0.0) and lower > 0.0),
    }


def _gate(
    rows: list[dict[str, Any]], config: GeometryGateExperimentConfig
) -> dict[str, Any]:
    indexed = {
        (
            row["training_protocol"],
            row["model_seed"],
            row["split"],
            row["method"],
            row["perturbation_bins"],
            row["family"],
            row["sample_id"],
        ): row
        for row in rows
    }
    comparisons = []
    for protocol_index, protocol in enumerate(TRAINING_PROTOCOLS):
        for split_index, split in enumerate(EVALUATION_SPLITS):
            cloud_keys = sorted(
                {
                    (row["family"], row["sample_id"])
                    for row in rows
                    if row["training_protocol"] == protocol and row["split"] == split
                }
            )
            families = np.asarray([key[0] for key in cloud_keys])
            for metric_index, metric in enumerate(("x_c_chamfer_mse", "surface_mse")):
                values = {}
                for method in METHODS:
                    values[method] = np.asarray(
                        [
                            [
                                indexed[
                                    (
                                        protocol,
                                        model_seed,
                                        split,
                                        method,
                                        0,
                                        *cloud,
                                    )
                                ][metric]
                                for cloud in cloud_keys
                            ]
                            for model_seed in config.model_seeds
                        ]
                    )
                comparisons.append(
                    {
                        "training_protocol": protocol,
                        "split": split,
                        "metric": metric,
                        **_paired_gate_comparison(
                            values["fps"],
                            values["refiner"],
                            families,
                            config=config,
                            seed=(
                                config.bootstrap_seed
                                + protocol_index * 10_000
                                + split_index * 100
                                + metric_index
                            ),
                        ),
                    }
                )
    return {
        "criterion": (
            "for both exact-sample and independent-resampling training, the "
            "unperturbed refiner must beat matched FPS on every model seed and "
            "the paired hierarchical 95% interval must exclude zero for fresh "
            "X_c Chamfer RMSE and analytic reconstruction-to-surface RMSE on "
            "validation and parameter OOD"
        ),
        "comparisons": comparisons,
        "passes": bool(comparisons and all(row["passes"] for row in comparisons)),
    }


def run_geometry_gate_experiment(
    config: GeometryGateExperimentConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train matched target protocols and evaluate the complete Gate B grid."""

    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_protocol = _data_protocol(config)
    if not data_protocol["all_role_samples_byte_distinct"]:
        raise RuntimeError("procedural X_a, X_b, and X_c draws are not independent")
    manifest = {
        "experiment": "031_procedural_geometry_gate",
        "config": json.loads(json.dumps(asdict(config))),
        "data_protocol": data_protocol,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    training_records = []
    rows = []
    for protocol in TRAINING_PROTOCOLS:
        datasets = _datasets(config, protocol)
        for model_seed in config.model_seeds:
            artifact_dir = output_dir / protocol / f"seed_{model_seed}"
            experiment_config = _refiner_config(
                config, model_seed=model_seed, output_dir=artifact_dir
            )
            training = run_refiner_experiment(
                experiment_config,
                device_name=str(device),
                loader_factory=_loader_factory(
                    datasets, batch_size=config.batch_size, seed=model_seed
                ),
                data_identity={
                    "training_protocol": protocol,
                    "data_protocol_sha256": hashlib.sha256(
                        json.dumps(data_protocol, sort_keys=True).encode()
                    ).hexdigest(),
                },
            )
            training_records.append(
                {
                    "training_protocol": protocol,
                    "model_seed": model_seed,
                    "decoder_hash": training["decoder_hash_after_refiner"],
                    "refiner_hash": training["refiner_hash_after_training"],
                    "decoder_unchanged": training["decoder_unchanged"],
                }
            )
            decoder, refiner = _models(config, artifact_dir, device)
            rows.extend(
                _evaluate_models(
                    config,
                    datasets,
                    training_protocol=protocol,
                    model_seed=model_seed,
                    decoder=decoder,
                    refiner=refiner,
                    device=device,
                )
            )
    with (output_dir / "geometry_per_cloud.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "experiment": "031_procedural_geometry_gate",
        "config": json.loads(json.dumps(asdict(config))),
        "device": str(device),
        "data_protocol": data_protocol,
        "training_records": training_records,
        "summary": _aggregate_rows(rows),
        "gate_b": _gate(rows, config),
        "contract_checks": {
            "independent_x_a_x_b_x_c": data_protocol["all_role_samples_byte_distinct"],
            "all_families_present_in_each_split": all(
                {record["family"] for record in split["records"]} == set(FAMILIES)
                for split in data_protocol["splits"].values()
            ),
            "decoder_frozen_during_refiner_training": all(
                record["decoder_unchanged"] for record in training_records
            ),
            "encoder_feedback_uses_x_a_only": True,
            "all_stream_round_trips_exact": all(
                row["serialized_round_trip_exact"] for row in rows
            ),
            "all_coordinates_on_exact_lattice": all(
                row["coordinates_on_exact_lattice"] for row in rows
            ),
            "both_training_protocols_present": {
                row["training_protocol"] for row in rows
            }
            == set(TRAINING_PROTOCOLS),
            "one_and_two_bin_perturbations_present": {
                row["perturbation_bins"] for row in rows
            }
            == {0, *config.perturbation_bins},
        },
        "per_cloud_rows": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "per_cloud_path": str(output_dir / "geometry_per_cloud.jsonl"),
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError("Experiment 031 procedural scientific contract failed")
    (output_dir / "geometry_gate_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_031_geometry_gate_smoke.json"),
    )
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = GeometryGateExperimentConfig.from_json(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    run_geometry_gate_experiment(config, device_name=args.device or "auto")


if __name__ == "__main__":
    main()
