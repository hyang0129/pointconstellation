"""Run bitstream-level procedural or manifest-backed compression benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pointconstellation.bitstream import (
    decode_constellation,
    encode_constellation,
    expected_stream_bytes,
)
from pointconstellation.codecs import Tmc3RatePoint, run_pc_error, run_tmc3
from pointconstellation.data import (
    MeshSurfaceDataset,
    file_sha256,
    generate_sample,
    load_mesh_manifest,
)
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.refiner import CompetitiveConstellationRefiner
from pointconstellation.refiner_experiment import (
    RefinerExperimentConfig,
    _fps,
    run_refiner_experiment,
)
from pointconstellation.standardized_metrics import standardized_geometry_metrics
from pointconstellation.train import select_device, set_seed

METHODS = ("fps", "free", "strict_subset")


@dataclass(frozen=True)
class GpccBenchmarkConfig:
    executable: str
    rate_points: tuple[Tmc3RatePoint, ...]
    encoder_args: tuple[str, ...] = ()
    position_bits: int = 12
    timeout_seconds: float = 120.0
    amortize_parameter_sets_over: int | None = None

    def __post_init__(self) -> None:
        if not self.rate_points:
            raise ValueError("G-PCC requires at least one rate point")
        if len({point.name for point in self.rate_points}) != len(self.rate_points):
            raise ValueError("G-PCC rate-point names must be unique")
        if not 2 <= self.position_bits <= 24:
            raise ValueError("G-PCC position_bits must be between 2 and 24")
        if self.timeout_seconds <= 0:
            raise ValueError("G-PCC timeout_seconds must be positive")
        if self.amortize_parameter_sets_over is not None and (
            not isinstance(self.amortize_parameter_sets_over, int)
            or isinstance(self.amortize_parameter_sets_over, bool)
            or self.amortize_parameter_sets_over < 1
        ):
            raise ValueError("amortize_parameter_sets_over must be a positive integer")
        Tmc3RatePoint("common", self.encoder_args)
        common_options = {argument.split("=", 1)[0] for argument in self.encoder_args}
        for point in self.rate_points:
            point_options = {
                argument.split("=", 1)[0] for argument in point.encoder_args
            }
            duplicate = common_options & point_options
            if duplicate:
                raise ValueError(
                    f"G-PCC common/rate arguments overlap: {sorted(duplicate)}"
                )


@dataclass(frozen=True)
class StandardizedBenchmarkConfig:
    """Configuration for training plus bitstream-level evaluation."""

    experiment: RefinerExperimentConfig
    profile_name: str = "macbook"
    evaluation_methods: tuple[str, ...] = METHODS
    distance_chunk_size: int = 256
    sliced_directions: int = 32
    peak_distance: float = 2.0
    dataset_kind: str = "procedural"
    dataset_root: str | None = None
    dataset_manifest: str | None = None
    mesh_ood_split: str = "category_ood"
    mesh_training_target: str = "source"
    verify_mesh_hashes: bool = True
    gpcc: GpccBenchmarkConfig | None = None
    official_metric_executable: str | None = None
    official_metric_position_bits: int = 12
    official_metric_timeout_seconds: float = 120.0
    output_dir: str = "artifacts/local/experiment_014_standardized_macbook"

    def __post_init__(self) -> None:
        if not self.evaluation_methods:
            raise ValueError("evaluation_methods must be nonempty")
        if any(method not in METHODS for method in self.evaluation_methods):
            raise ValueError(f"evaluation_methods must be selected from {METHODS}")
        if len(set(self.evaluation_methods)) != len(self.evaluation_methods):
            raise ValueError("evaluation_methods must be unique")
        if self.distance_chunk_size < 1 or self.sliced_directions < 1:
            raise ValueError("metric chunk size and directions must be positive")
        if self.peak_distance <= 0:
            raise ValueError("peak_distance must be positive")
        if self.experiment.input_sizes != (self.experiment.num_points,):
            raise ValueError("the standardized profile requires full-cloud inputs")
        if self.experiment.run_internal_evaluation:
            raise ValueError("disable the legacy evaluation for the standardized run")
        if self.dataset_kind not in {"procedural", "mesh_manifest"}:
            raise ValueError("dataset_kind must be procedural or mesh_manifest")
        if self.dataset_kind == "mesh_manifest" and (
            self.dataset_root is None or self.dataset_manifest is None
        ):
            raise ValueError("mesh_manifest datasets require root and manifest paths")
        if self.mesh_training_target not in {"source", "independent"}:
            raise ValueError("mesh_training_target must be source or independent")
        if not 2 <= self.official_metric_position_bits <= 24:
            raise ValueError("official_metric_position_bits must be between 2 and 24")
        if self.official_metric_timeout_seconds <= 0:
            raise ValueError("official_metric_timeout_seconds must be positive")

    @classmethod
    def from_json(cls, path: Path) -> StandardizedBenchmarkConfig:
        values = json.loads(path.read_text())
        experiment_values = values.pop("experiment")
        gpcc_values = values.pop("gpcc", None)
        for key in ("input_sizes", "constellation_sizes"):
            if key in experiment_values:
                experiment_values[key] = tuple(experiment_values[key])
        if "evaluation_methods" in values:
            values["evaluation_methods"] = tuple(values["evaluation_methods"])
        gpcc = None
        if gpcc_values is not None:
            if "encoder_args" in gpcc_values:
                gpcc_values["encoder_args"] = tuple(gpcc_values["encoder_args"])
            gpcc_values["rate_points"] = tuple(
                Tmc3RatePoint(
                    name=point["name"], encoder_args=tuple(point["encoder_args"])
                )
                for point in gpcc_values["rate_points"]
            )
            gpcc = GpccBenchmarkConfig(**gpcc_values)
        return cls(
            experiment=RefinerExperimentConfig(**experiment_values),
            gpcc=gpcc,
            **values,
        )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed(device: torch.device, operation: Any) -> tuple[Any, float]:
    _sync(device)
    started = time.perf_counter()
    result = operation()
    _sync(device)
    return result, time.perf_counter() - started


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _load_models(
    config: RefinerExperimentConfig, device: torch.device
) -> tuple[VariableConstellationDecoder, CompetitiveConstellationRefiner]:
    output_dir = Path(config.output_dir)
    decoder = VariableConstellationDecoder(
        config.num_points,
        max(config.constellation_sizes),
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    refiner = CompetitiveConstellationRefiner(
        max(config.constellation_sizes),
        bits=config.bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        recurrent_steps=config.recurrent_steps,
        responsibility_temperature=config.responsibility_temperature,
        maximum_update=config.maximum_update,
        use_decoder_gradient=config.use_decoder_gradient,
        decoder_gradient_chunk_size=config.decoder_gradient_chunk_size,
    ).to(device)
    decoder_checkpoint = torch.load(
        output_dir / "decoder.pt", map_location=device, weights_only=True
    )
    refiner_checkpoint = torch.load(
        output_dir / "refiner.pt", map_location=device, weights_only=True
    )
    decoder.load_state_dict(decoder_checkpoint["model"])
    refiner.load_state_dict(refiner_checkpoint["model"])
    return decoder.eval().requires_grad_(False), refiner.eval().requires_grad_(False)


def _decode_and_reconstruct(
    stream: bytes,
    *,
    device: torch.device,
    dtype: torch.dtype,
    decoder: VariableConstellationDecoder,
) -> tuple[Any, Tensor]:
    packet = decode_constellation(stream)
    decoded = torch.from_numpy(packet.coordinates).to(device=device, dtype=dtype)[None]
    reconstruction = decoder(decoded, num_output_points=packet.output_points)
    return packet, reconstruction


def _manifest_hash(*, split: str, samples: int, num_points: int, seed: int) -> str:
    digest = hashlib.sha256()
    for sample_id in range(samples):
        sample = generate_sample(
            sample_id, num_points=num_points, seed=seed, split=split
        )
        digest.update(sample.family.encode())
        digest.update(sample.points.tobytes())
        digest.update(sample.normals.tobytes())
    return digest.hexdigest()


def _mesh_datasets(
    config: StandardizedBenchmarkConfig, experiment: RefinerExperimentConfig
) -> dict[str, MeshSurfaceDataset]:
    assert config.dataset_root is not None and config.dataset_manifest is not None
    root = Path(config.dataset_root)
    manifest = Path(config.dataset_manifest)
    data_seed = (
        experiment.seed if experiment.data_seed is None else experiment.data_seed
    )
    datasets = {
        split: MeshSurfaceDataset(
            root,
            manifest,
            split=split,
            num_points=experiment.num_points,
            seed=data_seed,
            verify_hashes=config.verify_mesh_hashes,
            training_target=config.mesh_training_target,
        )
        for split in ("train", "validation", config.mesh_ood_split)
    }
    expected = {
        "train": experiment.train_samples,
        "validation": experiment.validation_samples,
        config.mesh_ood_split: experiment.parameter_ood_samples,
    }
    for split, size in expected.items():
        if len(datasets[split]) != size:
            raise ValueError(
                f"manifest split {split} has {len(datasets[split])} records; "
                f"experiment expects {size}"
            )
    return datasets


def _mesh_loader_factory(
    config: StandardizedBenchmarkConfig, experiment: RefinerExperimentConfig
) -> Any:
    def make_loaders() -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
        datasets = _mesh_datasets(config, experiment)
        generator = torch.Generator().manual_seed(experiment.seed)
        return (
            DataLoader(
                datasets["train"],
                batch_size=experiment.batch_size,
                shuffle=True,
                num_workers=0,
                generator=generator,
            ),
            DataLoader(
                datasets["validation"],
                batch_size=experiment.batch_size,
                shuffle=False,
                num_workers=0,
            ),
            DataLoader(
                datasets[config.mesh_ood_split],
                batch_size=experiment.batch_size,
                shuffle=False,
                num_workers=0,
            ),
        )

    return make_loaders


def _data_identity(
    config: StandardizedBenchmarkConfig, experiment: RefinerExperimentConfig
) -> dict[str, Any]:
    if config.dataset_kind == "procedural":
        return {
            "kind": "procedural",
            "seed": (
                experiment.seed
                if experiment.data_seed is None
                else experiment.data_seed
            ),
            "train_samples": experiment.train_samples,
        }
    assert config.dataset_manifest is not None
    manifest_path = Path(config.dataset_manifest)
    manifest = load_mesh_manifest(manifest_path)
    return {
        "kind": "mesh_manifest",
        "dataset": manifest["dataset"],
        "source": manifest.get("source"),
        "manifest_sha256": file_sha256(manifest_path),
        "verify_mesh_hashes": config.verify_mesh_hashes,
        "training_target": config.mesh_training_target,
    }


def _constellation_for_method(
    source: Tensor,
    *,
    method: str,
    size: int,
    bits: int,
    decoder: VariableConstellationDecoder,
    refiner: CompetitiveConstellationRefiner,
    output_points: int,
) -> Tensor:
    if method == "fps":
        return _fps(source, size, bits)
    free = refiner(
        source,
        size,
        decoder=decoder,
        target=source,
        num_output_points=output_points,
    )
    if method == "free":
        return free
    if method == "strict_subset":
        return refiner.project_unique_to_input(free, source)
    raise ValueError(f"unknown method: {method}")


def _average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["split"], row["method"], row["constellation_size"])
        groups.setdefault(key, []).append(row)
    summaries = []
    scalar_fields = (
        "nominal_payload_bpp",
        "payload_bpp",
        "actual_stream_bpp",
        "header_bytes",
        "payload_bytes",
        "stream_bytes",
        "encoder_inference_seconds",
        "bitstream_encode_seconds",
        "encode_seconds",
        "decode_seconds",
        "chamfer_mse",
        "chamfer_rmse",
        "d1_mse_proxy",
        "d1_psnr_db_proxy",
        "d2_mse_proxy",
        "d2_psnr_db_proxy",
        "p95_euclidean",
        "p99_euclidean",
        "hausdorff",
        "sliced_wasserstein_rms_proxy",
    )
    if rows and "fresh_chamfer_mse" in rows[0]:
        scalar_fields += tuple(
            f"fresh_{field}"
            for field in (
                "chamfer_mse",
                "chamfer_rmse",
                "d1_mse_proxy",
                "d1_psnr_db_proxy",
                "d2_mse_proxy",
                "d2_psnr_db_proxy",
                "p95_euclidean",
                "p99_euclidean",
                "hausdorff",
                "sliced_wasserstein_rms_proxy",
            )
        )
    if rows and "official_d1_mse" in rows[0]:
        scalar_fields += tuple(
            f"official_{field}"
            for field in (
                "d1_mse",
                "d1_psnr_db",
                "d2_mse",
                "d2_psnr_db",
                "d1_hausdorff",
                "d1_hausdorff_psnr_db",
                "d2_hausdorff",
                "d2_hausdorff_psnr_db",
                "elapsed_seconds",
            )
        )
    for (split, method, size), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "split": split,
            "method": method,
            "constellation_size": size,
            "samples": len(group),
        }
        for field in scalar_fields:
            values = np.asarray([row[field] for row in group], dtype=np.float64)
            summary[field] = float(values.mean())
        summaries.append(summary)
    return summaries


def _monotonicity(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary:
        groups.setdefault((row["split"], row["method"]), []).append(row)
    results = []
    for (split, method), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: row["actual_stream_bpp"])
        violations = []
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current["chamfer_rmse"] > previous["chamfer_rmse"] + 1e-12:
                violations.append(
                    {
                        "from_k": previous["constellation_size"],
                        "to_k": current["constellation_size"],
                        "rmse_change": current["chamfer_rmse"]
                        - previous["chamfer_rmse"],
                    }
                )
        results.append(
            {
                "split": split,
                "method": method,
                "rate_points": len(ordered),
                "chamfer_rmse_nonincreasing": not violations,
                "violations": violations,
            }
        )
    return results


def _average_gpcc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["split"], row["rate_point"]), []).append(row)
    excluded = {
        "split",
        "sample_id",
        "family",
        "model_id",
        "method",
        "rate_point",
        "encoder_args",
    }
    summaries = []
    for (split, rate_point), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "split": split,
            "method": "gpcc_octree",
            "rate_point": rate_point,
            "samples": len(group),
            "encoder_args": group[0]["encoder_args"],
        }
        for field in sorted(group[0].keys() - excluded):
            if isinstance(group[0][field], int | float):
                values = np.asarray([row[field] for row in group], dtype=np.float64)
                summary[field] = float(values.mean())
        summaries.append(summary)
    return summaries


def _gpcc_monotonicity(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for split in sorted({row["split"] for row in summary}):
        ordered = sorted(
            (row for row in summary if row["split"] == split),
            key=lambda row: row["actual_stream_bpp"],
        )
        violations = []
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current["chamfer_rmse"] > previous["chamfer_rmse"] + 1e-12:
                violations.append(
                    {
                        "from_rate_point": previous["rate_point"],
                        "to_rate_point": current["rate_point"],
                        "rmse_change": current["chamfer_rmse"]
                        - previous["chamfer_rmse"],
                    }
                )
        results.append(
            {
                "split": split,
                "method": "gpcc_octree",
                "rate_points": len(ordered),
                "chamfer_rmse_nonincreasing": not violations,
                "violations": violations,
            }
        )
    return results


def _pareto_frontier(
    rows: list[dict[str, Any]],
    *,
    rate_field: str = "actual_stream_bpp",
    distortion_field: str = "chamfer_rmse",
) -> list[dict[str, Any]]:
    """Return non-dominated rate-distortion rows for every evaluation split."""

    frontier = []
    for split in sorted({row["split"] for row in rows}):
        candidates = [row for row in rows if row["split"] == split]
        for row in candidates:
            dominated = any(
                other[rate_field] <= row[rate_field]
                and other[distortion_field] <= row[distortion_field]
                and (
                    other[rate_field] < row[rate_field]
                    or other[distortion_field] < row[distortion_field]
                )
                for other in candidates
            )
            if not dominated:
                frontier.append(row)
    return sorted(frontier, key=lambda row: (row["split"], row[rate_field]))


def _interpolation_free_gpcc_comparisons(
    neural_summary: list[dict[str, Any]], gpcc_summary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare measured points without interpolating between codec rates."""

    comparisons = []
    for neural in neural_summary:
        candidates = [row for row in gpcc_summary if row["split"] == neural["split"]]
        at_or_below_rate = [
            row
            for row in candidates
            if row["actual_stream_bpp"] <= neural["actual_stream_bpp"]
        ]
        at_or_below_distortion = [
            row for row in candidates if row["chamfer_rmse"] <= neural["chamfer_rmse"]
        ]
        best_below_rate = (
            min(at_or_below_rate, key=lambda row: row["chamfer_rmse"])
            if at_or_below_rate
            else None
        )
        lowest_below_distortion = (
            min(
                at_or_below_distortion,
                key=lambda row: row["actual_stream_bpp"],
            )
            if at_or_below_distortion
            else None
        )

        def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
            if row is None:
                return None
            return {
                "rate_point": row["rate_point"],
                "actual_stream_bpp": row["actual_stream_bpp"],
                "chamfer_rmse": row["chamfer_rmse"],
            }

        comparisons.append(
            {
                "split": neural["split"],
                "method": neural["method"],
                "constellation_size": neural["constellation_size"],
                "actual_stream_bpp": neural["actual_stream_bpp"],
                "chamfer_rmse": neural["chamfer_rmse"],
                "best_gpcc_at_or_below_rate": compact(best_below_rate),
                "lowest_rate_gpcc_at_or_below_distortion": compact(
                    lowest_below_distortion
                ),
                "gpcc_measured_point_dominates": any(
                    row["actual_stream_bpp"] <= neural["actual_stream_bpp"]
                    and row["chamfer_rmse"] <= neural["chamfer_rmse"]
                    for row in candidates
                ),
            }
        )
    return comparisons


def _matching_model_run(
    config: RefinerExperimentConfig, data_identity: dict[str, Any]
) -> dict[str, Any] | None:
    metrics_path = Path(config.output_dir) / "metrics.json"
    checkpoints = (
        Path(config.output_dir) / "decoder.pt",
        Path(config.output_dir) / "refiner.pt",
    )
    if not metrics_path.exists() or not all(path.exists() for path in checkpoints):
        return None
    result = json.loads(metrics_path.read_text())
    normalized_config = json.loads(json.dumps(asdict(config)))
    matches = result.get("config") == normalized_config
    matches = matches and result.get("data_identity") == data_identity
    return result if matches else None


def run_standardized_benchmark(
    config: StandardizedBenchmarkConfig,
    *,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train the codec and evaluate its real fixed-width streams."""

    set_seed(config.experiment.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = replace(
        config.experiment,
        output_dir=str(output_dir / "model"),
        run_internal_evaluation=False,
    )
    data_identity = _data_identity(config, experiment)
    loader_factory = (
        _mesh_loader_factory(config, experiment)
        if config.dataset_kind == "mesh_manifest"
        else None
    )
    model_result = _matching_model_run(experiment, data_identity) if resume else None
    model_reused = model_result is not None
    if model_result is None:
        model_result = run_refiner_experiment(
            experiment,
            device_name=str(device),
            loader_factory=loader_factory,
            data_identity=data_identity,
        )
    decoder, refiner = _load_models(experiment, device)
    data_seed = (
        experiment.seed if experiment.data_seed is None else experiment.data_seed
    )

    rows: list[dict[str, Any]] = []
    gpcc_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    ood_label = (
        "parameter_ood"
        if config.dataset_kind == "procedural"
        else config.mesh_ood_split
    )
    split_sizes = {
        "validation": experiment.validation_samples,
        ood_label: experiment.parameter_ood_samples,
    }
    evaluation_meshes = (
        _mesh_datasets(config, experiment)
        if config.dataset_kind == "mesh_manifest"
        else None
    )
    with torch.no_grad():
        for split, samples in split_sizes.items():
            for sample_id in range(samples):
                if evaluation_meshes is None:
                    sample = generate_sample(
                        sample_id,
                        num_points=experiment.num_points,
                        seed=data_seed,
                        split=split,
                    )
                    source_cpu = torch.from_numpy(sample.points)
                    target_cpu = source_cpu
                    normals_cpu = torch.from_numpy(sample.normals)
                    family = sample.family
                    model_id = str(sample.sample_id)
                else:
                    mesh_sample = evaluation_meshes[split].sample(sample_id)
                    source_cpu = torch.from_numpy(mesh_sample.source_points)
                    target_cpu = source_cpu
                    normals_cpu = torch.from_numpy(mesh_sample.source_normals)
                    fresh_target_cpu = torch.from_numpy(mesh_sample.target_points)
                    fresh_normals_cpu = torch.from_numpy(mesh_sample.target_normals)
                    family = mesh_sample.category
                    model_id = mesh_sample.model_id
                source = source_cpu.to(device)[None]
                for size in experiment.constellation_sizes:
                    for method in config.evaluation_methods:
                        constellation, encode_seconds = _timed(
                            device,
                            lambda source=source, size=size, method=method: (
                                _constellation_for_method(
                                    source,
                                    method=method,
                                    size=size,
                                    bits=experiment.bits,
                                    decoder=decoder,
                                    refiner=refiner,
                                    output_points=experiment.num_points,
                                )
                            ),
                        )
                        serialization_started = time.perf_counter()
                        stream = encode_constellation(
                            constellation[0].detach().cpu().numpy(),
                            bits=experiment.bits,
                            mode=method,
                            output_points=experiment.num_points,
                        )
                        bitstream_encode_seconds = (
                            time.perf_counter() - serialization_started
                        )
                        (packet, reconstruction), decode_seconds = _timed(
                            device,
                            lambda stream=stream, dtype=source.dtype: (
                                _decode_and_reconstruct(
                                    stream,
                                    device=device,
                                    dtype=dtype,
                                    decoder=decoder,
                                )
                            ),
                        )
                        metrics = standardized_geometry_metrics(
                            reconstruction[0].cpu(),
                            target_cpu,
                            normals_cpu,
                            chunk_size=config.distance_chunk_size,
                            sliced_directions=config.sliced_directions,
                            peak_distance=config.peak_distance,
                        )
                        if evaluation_meshes is not None:
                            fresh_metrics = standardized_geometry_metrics(
                                reconstruction[0].cpu(),
                                fresh_target_cpu,
                                fresh_normals_cpu,
                                chunk_size=config.distance_chunk_size,
                                sliced_directions=config.sliced_directions,
                                peak_distance=config.peak_distance,
                            )
                            metrics.update(
                                {
                                    f"fresh_{name}": value
                                    for name, value in fresh_metrics.items()
                                }
                            )
                        if config.official_metric_executable is not None:
                            official = run_pc_error(
                                Path(config.official_metric_executable),
                                source_cpu.numpy(),
                                reconstruction[0].cpu().numpy(),
                                normals_cpu.numpy(),
                                work_dir=(
                                    output_dir
                                    / "official_metric_work"
                                    / split
                                    / f"sample_{sample_id:05d}"
                                    / method
                                    / f"k_{size:04d}"
                                ),
                                position_bits=config.official_metric_position_bits,
                                timeout_seconds=config.official_metric_timeout_seconds,
                            )
                            metrics.update(
                                {
                                    f"official_{name}": value
                                    for name, value in official.metrics.items()
                                }
                            )
                            metrics["official_elapsed_seconds"] = (
                                official.elapsed_seconds
                            )
                        rows.append(
                            {
                                "split": split,
                                "sample_id": sample_id,
                                "family": family,
                                "model_id": model_id,
                                "method": method,
                                "constellation_size": size,
                                "coordinate_bits": experiment.bits,
                                "payload_bits": packet.payload_bits,
                                "nominal_payload_bpp": packet.payload_bits
                                / experiment.num_points,
                                "header_bytes": packet.header_bytes,
                                "payload_bytes": packet.payload_bytes,
                                "payload_bpp": packet.payload_bytes
                                * 8
                                / experiment.num_points,
                                "stream_bytes": packet.stream_bytes,
                                "actual_stream_bpp": packet.stream_bytes
                                * 8
                                / experiment.num_points,
                                "encoder_inference_seconds": encode_seconds,
                                "bitstream_encode_seconds": bitstream_encode_seconds,
                                "encode_seconds": encode_seconds
                                + bitstream_encode_seconds,
                                "decode_seconds": decode_seconds,
                                **metrics,
                            }
                        )
                if config.gpcc is not None:
                    for rate_point in config.gpcc.rate_points:
                        effective_rate_point = Tmc3RatePoint(
                            rate_point.name,
                            (*config.gpcc.encoder_args, *rate_point.encoder_args),
                        )
                        gpcc_result = run_tmc3(
                            Path(config.gpcc.executable),
                            source_cpu.numpy(),
                            rate_point=effective_rate_point,
                            work_dir=(
                                output_dir
                                / "gpcc_work"
                                / split
                                / f"sample_{sample_id:05d}"
                                / rate_point.name
                            ),
                            position_bits=config.gpcc.position_bits,
                            timeout_seconds=config.gpcc.timeout_seconds,
                        )
                        gpcc_reconstruction = torch.from_numpy(
                            gpcc_result.reconstruction
                        )
                        gpcc_metrics = standardized_geometry_metrics(
                            gpcc_reconstruction,
                            target_cpu,
                            normals_cpu,
                            chunk_size=config.distance_chunk_size,
                            sliced_directions=config.sliced_directions,
                            peak_distance=config.peak_distance,
                        )
                        if evaluation_meshes is not None:
                            fresh_gpcc_metrics = standardized_geometry_metrics(
                                gpcc_reconstruction,
                                fresh_target_cpu,
                                fresh_normals_cpu,
                                chunk_size=config.distance_chunk_size,
                                sliced_directions=config.sliced_directions,
                                peak_distance=config.peak_distance,
                            )
                            gpcc_metrics.update(
                                {
                                    f"fresh_{name}": value
                                    for name, value in fresh_gpcc_metrics.items()
                                }
                            )
                        if config.official_metric_executable is not None:
                            official = run_pc_error(
                                Path(config.official_metric_executable),
                                source_cpu.numpy(),
                                gpcc_result.reconstruction,
                                normals_cpu.numpy(),
                                work_dir=(
                                    output_dir
                                    / "official_metric_work"
                                    / split
                                    / f"sample_{sample_id:05d}"
                                    / "gpcc_octree"
                                    / rate_point.name
                                ),
                                position_bits=config.official_metric_position_bits,
                                timeout_seconds=config.official_metric_timeout_seconds,
                            )
                            gpcc_metrics.update(
                                {
                                    f"official_{name}": value
                                    for name, value in official.metrics.items()
                                }
                            )
                            gpcc_metrics["official_elapsed_seconds"] = (
                                official.elapsed_seconds
                            )
                        breakdown = gpcc_result.stream_breakdown
                        gpcc_row = {
                            "split": split,
                            "sample_id": sample_id,
                            "family": family,
                            "model_id": model_id,
                            "method": "gpcc_octree",
                            "rate_point": rate_point.name,
                            "encoder_args": list(effective_rate_point.encoder_args),
                            "sps_bytes": breakdown.sps_bytes,
                            "gps_bytes": breakdown.gps_bytes,
                            "slice_header_bytes": breakdown.slice_header_bytes,
                            "header_bytes": breakdown.header_bytes,
                            "payload_bytes": breakdown.payload_bytes,
                            "payload_bpp": breakdown.payload_bytes
                            * 8
                            / experiment.num_points,
                            "stream_bytes": gpcc_result.stream_bytes,
                            "actual_stream_bpp": gpcc_result.stream_bytes
                            * 8
                            / experiment.num_points,
                            "reconstruction_points": len(gpcc_result.reconstruction),
                            "encode_seconds": gpcc_result.encode_seconds,
                            "decode_seconds": gpcc_result.decode_seconds,
                            **gpcc_metrics,
                        }
                        if config.gpcc.amortize_parameter_sets_over is not None:
                            period = config.gpcc.amortize_parameter_sets_over
                            amortized_bytes = breakdown.amortized_stream_bytes(period)
                            gpcc_row.update(
                                {
                                    "amortize_parameter_sets_over": period,
                                    "amortized_stream_bytes": amortized_bytes,
                                    "amortized_stream_bpp": amortized_bytes
                                    * 8
                                    / experiment.num_points,
                                    "amortized_stream_note": (
                                        "accounting_only_not_a_decodable_stream"
                                    ),
                                }
                            )
                        gpcc_rows.append(gpcc_row)

    summary = _average_rows(rows)
    model_dir = Path(experiment.output_dir)
    if evaluation_meshes is None:
        manifests = {
            split: {
                "samples": samples,
                "sha256": _manifest_hash(
                    split=split,
                    samples=samples,
                    num_points=experiment.num_points,
                    seed=data_seed,
                ),
            }
            for split, samples in split_sizes.items()
        }
    else:
        assert config.dataset_manifest is not None
        manifest = load_mesh_manifest(Path(config.dataset_manifest))
        manifests = {
            "dataset": manifest["dataset"],
            "source": manifest.get("source"),
            "sampling": manifest.get("sampling"),
            "categories": manifest.get("categories"),
            "manifest_path": config.dataset_manifest,
            "manifest_sha256": file_sha256(Path(config.dataset_manifest)),
            "splits": {
                split: {"samples": samples} for split, samples in split_sizes.items()
            },
        }
    gpcc_summary = _average_gpcc_rows(gpcc_rows)
    gpcc_frontier = _pareto_frontier(gpcc_summary)
    gpcc_metadata = None
    if config.gpcc is not None:
        executable = Path(config.gpcc.executable)
        gpcc_metadata = {
            "executable": str(executable),
            "executable_sha256": file_sha256(executable),
            "position_bits": config.gpcc.position_bits,
            "common_encoder_args": list(config.gpcc.encoder_args),
            "amortize_parameter_sets_over": (config.gpcc.amortize_parameter_sets_over),
            "amortized_stream_note": (
                "SPS/GPS accounting only; reported bytes are not a decodable stream"
                if config.gpcc.amortize_parameter_sets_over is not None
                else None
            ),
            "summary": gpcc_summary,
            "pareto_frontier": gpcc_frontier,
            "monotonicity": _gpcc_monotonicity(gpcc_summary),
            "per_cloud": gpcc_rows,
        }
    result = {
        "protocol": {
            "name": (
                "pointconstellation-procedural-low-rate-v1"
                if config.dataset_kind == "procedural"
                else (
                    "pointconstellation-modelnet40-surface-pilot-v1"
                    if data_identity.get("dataset") == "ModelNet40"
                    else "pointconstellation-mesh-surface-pilot-v1"
                )
            ),
            "profile": config.profile_name,
            "data_identity": data_identity,
            "status": (
                "protocol_aligned_procedural_proxy"
                if config.dataset_kind == "procedural"
                else "external_mesh_surface_pilot"
            ),
            "normalized_domain": [-1.0, 1.0],
            "input_points": experiment.num_points,
            "coordinate_bits": experiment.bits,
            "constellation_sizes": list(experiment.constellation_sizes),
            "fixed_header_bytes": expected_stream_bytes(1, experiment.bits)
            - ((3 * experiment.bits + 7) // 8),
            "rate_definition": "total serialized stream bits / input points",
            "payload_rate_definition": (
                "byte-aligned payload bits / input points; outer format headers "
                "excluded"
            ),
            "metric_note": (
                "Unprefixed D1/D2 proxy fields use the declared normalized peak; "
                "sliced Wasserstein is an EMD proxy. official_* fields are emitted "
                "only when MPEG pc_error is configured."
            ),
            "comparability_claims": {
                "shapenet": False,
                "modelnet40": False,
                "mpeg_common_test_conditions": False,
                "official_pc_error": config.official_metric_executable is not None,
            },
            "official_dataset_source": data_identity.get("dataset"),
            "pilot_subset_not_full_benchmark": config.dataset_kind != "procedural",
        },
        "config": {
            **asdict(config),
            "experiment": asdict(experiment),
        },
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
        "manifests": manifests,
        "model": {
            "decoder_parameters": model_result["decoder_parameter_count"],
            "refiner_parameters": model_result["refiner_parameter_count"],
            "decoder_checkpoint_bytes": (model_dir / "decoder.pt").stat().st_size,
            "refiner_checkpoint_bytes": (model_dir / "refiner.pt").stat().st_size,
            "decoder_unchanged": model_result["decoder_unchanged"],
            "decoder_state_hash": model_result["decoder_hash_after_refiner"],
            "refiner_state_hash": model_result.get("refiner_hash_after_training"),
            "reused_matching_checkpoint": model_reused,
            "training_elapsed_seconds": model_result["elapsed_seconds"],
        },
        "summary": summary,
        "monotonicity": _monotonicity(summary),
        "per_cloud": rows,
        "gpcc": gpcc_metadata,
        "interpolation_free_gpcc_comparisons": (
            _interpolation_free_gpcc_comparisons(summary, gpcc_summary)
            if gpcc_summary
            else []
        ),
        "official_metric": (
            {
                "executable": config.official_metric_executable,
                "executable_sha256": file_sha256(
                    Path(config.official_metric_executable)
                ),
                "position_bits": config.official_metric_position_bits,
            }
            if config.official_metric_executable is not None
            else None
        ),
        "evaluation_elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": _peak_rss_bytes(),
    }
    (output_dir / "benchmark_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (output_dir / "per_cloud.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if gpcc_rows:
        with (output_dir / "gpcc_per_cloud.jsonl").open("w") as handle:
            for row in gpcc_rows:
                handle.write(json.dumps(row) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_014_standardized_macbook.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--tmc3")
    parser.add_argument("--pc-error")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = StandardizedBenchmarkConfig.from_json(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.tmc3 is not None:
        if config.gpcc is None:
            raise ValueError("--tmc3 requires a config with a gpcc section")
        config = replace(config, gpcc=replace(config.gpcc, executable=args.tmc3))
    if args.pc_error is not None:
        config = replace(config, official_metric_executable=args.pc_error)
    result = run_standardized_benchmark(
        config, device_name=args.device, resume=args.resume
    )
    print(
        json.dumps(
            {
                "device": result["device"],
                "profile": result["protocol"]["profile"],
                "rate_points": len(result["summary"]),
                "evaluation_elapsed_seconds": result["evaluation_elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "benchmark_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
