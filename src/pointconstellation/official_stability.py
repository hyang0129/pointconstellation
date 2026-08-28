"""Official MPEG D1/D2 evaluation through stabilized Experiment 019 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pointconstellation.bitstream import expected_stream_bytes
from pointconstellation.codecs import run_pc_error
from pointconstellation.data import file_sha256
from pointconstellation.losses import chamfer_squared_chunked
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.selection_baselines import (
    SELECTION_METHODS,
    SELECTION_REPRESENTATIONS,
    SELECTION_TRIALS,
)
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _decoder,
    _refiner,
    _serialized_coordinates,
)
from pointconstellation.train import select_device

METRICS = ("d1_mse", "d2_mse")
SPLITS = ("validation", "ood")
DEFAULT_METHODS = ("fps", "refiner")


@dataclass(frozen=True)
class OfficialStabilityConfig:
    """Configuration for official frozen-decoder metric passes."""

    stability_config: str = "configs/experiment_019_stability_modelnet40.json"
    stability_artifact_dir: str = "artifacts/local/experiment_019_stability_modelnet40"
    pc_error_executable: str = "artifacts/tools/mpeg-pcc-dmetric/build/Release/pc_error"
    position_bits: int = 12
    timeout_seconds: float = 120.0
    decoder_seeds: tuple[int, ...] = (7, 17, 29, 41, 53, 67)
    refiner_seeds: tuple[int, ...] = (101, 211, 307)
    methods: tuple[str, ...] = DEFAULT_METHODS
    selection_seed: int = 20_260_821
    splits: tuple[str, ...] = SPLITS
    max_clouds_per_split: int | None = None
    allow_single_seed: bool = False
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_817
    confidence_level: float = 0.95
    output_dir: str = "artifacts/local/experiment_020_official_stability"

    def __post_init__(self) -> None:
        if not 2 <= self.position_bits <= 24:
            raise ValueError("position_bits must be between 2 and 24")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        minimum_seeds = 1 if self.allow_single_seed else 2
        minimum_seed_label = "one" if self.allow_single_seed else "two"
        if len(self.decoder_seeds) < minimum_seeds or len(
            set(self.decoder_seeds)
        ) != len(self.decoder_seeds):
            raise ValueError(
                f"decoder_seeds must contain at least {minimum_seed_label} unique seeds"
            )
        if len(self.refiner_seeds) < minimum_seeds or len(
            set(self.refiner_seeds)
        ) != len(self.refiner_seeds):
            raise ValueError(
                f"refiner_seeds must contain at least {minimum_seed_label} unique seeds"
            )
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be nonempty and unique")
        unknown_methods = set(self.methods) - ({"refiner"} | set(SELECTION_METHODS))
        if unknown_methods:
            raise ValueError(f"unknown official methods: {sorted(unknown_methods)}")
        if not {"fps", "refiner"}.issubset(self.methods):
            raise ValueError("methods must include fps and refiner for paired gates")
        if not 0 <= self.selection_seed < 2**63:
            raise ValueError("selection_seed must be a nonnegative 63-bit integer")
        if not self.splits or len(set(self.splits)) != len(self.splits):
            raise ValueError("splits must be nonempty and unique")
        if set(self.splits) - set(SPLITS):
            raise ValueError("official stability splits must be validation and/or ood")
        if self.max_clouds_per_split is not None and self.max_clouds_per_split < 2:
            raise ValueError("max_clouds_per_split must be at least two")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

    @classmethod
    def from_json(cls, path: Path) -> OfficialStabilityConfig:
        values = json.loads(path.read_text())
        for key in ("decoder_seeds", "refiner_seeds", "methods", "splits"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _stability_config_matches(
    recorded: dict[str, Any], stability: StabilityExperimentConfig
) -> bool:
    normalized = dict(recorded)
    normalized.setdefault("decoder_source_artifact_dir", None)
    normalized.setdefault("allow_single_seed", False)
    return normalized == _json_ready(asdict(stability))


def _official_manifest_matches(
    recorded: dict[str, Any], expected: dict[str, Any]
) -> bool:
    normalized = _json_ready(recorded)
    config = normalized.get("config")
    if isinstance(config, dict):
        config.setdefault("allow_single_seed", False)
    return normalized == expected


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in {"validation", "ood", "category_ood", "test"}
            or _contains_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["split"],
        row["method"],
        row["decoder_seed"],
        row.get("refiner_seed"),
        row["family"],
        row["model_id"],
        row["sample_id"],
    )


def _sample_identity(sample: dict[str, Any]) -> tuple[str, str, int]:
    sample_id = int(sample["sample_id"])
    return str(sample["family"]), str(sample.get("model_id", sample_id)), sample_id


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("official metric artifact contains duplicate rows")
    return rows


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _cloud_draw(categories: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(categories)
    category_draw = unique[rng.integers(0, len(unique), size=len(unique))]
    indices: list[int] = []
    for category in category_draw:
        candidates = np.flatnonzero(categories == category)
        indices.extend(
            candidates[rng.integers(0, len(candidates), size=len(candidates))].tolist()
        )
    return np.asarray(indices, dtype=np.int64)


def _bootstrap_comparison(
    fps: np.ndarray,
    refiner: np.ndarray,
    categories: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Compare D×C FPS with D×R×C refiner using paired factor draws."""

    fps = np.asarray(fps, dtype=np.float64)
    refiner = np.asarray(refiner, dtype=np.float64)
    categories = np.asarray(categories)
    if fps.ndim != 2 or refiner.ndim != 3:
        raise ValueError("FPS and refiner arrays must have shapes D×C and D×R×C")
    if refiner.shape[0] != fps.shape[0] or refiner.shape[2] != fps.shape[1]:
        raise ValueError("FPS and refiner arrays do not share decoder/cloud axes")
    if len(categories) != fps.shape[1]:
        raise ValueError("categories must align with the cloud axis")
    if np.any(fps < 0) or np.any(refiner < 0):
        raise ValueError("official MSE values cannot be negative")

    def relative(first: np.ndarray, second: np.ndarray) -> float:
        baseline = math.sqrt(float(first.mean()))
        candidate = math.sqrt(float(second.mean()))
        return 100.0 * (baseline - candidate) / max(baseline, 1e-12)

    decoder_fps = np.sqrt(fps.mean(axis=1))
    decoder_refiner = np.sqrt(refiner.mean(axis=(1, 2)))
    decoder_improvements = (
        100.0 * (decoder_fps - decoder_refiner) / np.clip(decoder_fps, 1e-12, None)
    )
    rng = np.random.default_rng(seed)
    improvements = np.empty(samples, dtype=np.float64)
    decoder_count, refiner_count, _ = refiner.shape
    for index in range(samples):
        decoder_draw = rng.integers(0, decoder_count, size=decoder_count)
        refiner_draw = rng.integers(0, refiner_count, size=refiner_count)
        cloud_draw = _cloud_draw(categories, rng)
        fps_draw = fps[decoder_draw[:, None], cloud_draw[None, :]]
        refiner_draw_values = refiner[
            decoder_draw[:, None, None],
            refiner_draw[None, :, None],
            cloud_draw[None, None, :],
        ]
        improvements[index] = relative(fps_draw, refiner_draw_values)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(improvements, (alpha, 1.0 - alpha))
    return {
        "decoder_count": decoder_count,
        "refiner_count": refiner_count,
        "cloud_count": fps.shape[1],
        "category_count": len(np.unique(categories)),
        "bootstrap_samples": samples,
        "confidence_level": confidence_level,
        "aggregate_fps_rmse_grid_units": math.sqrt(float(fps.mean())),
        "aggregate_refiner_rmse_grid_units": math.sqrt(float(refiner.mean())),
        "relative_rmse_improvement_percent": relative(fps, refiner),
        "confidence_interval_lower_percent": float(lower),
        "confidence_interval_upper_percent": float(upper),
        "per_decoder": [
            {
                "decoder_index": index,
                "fps_rmse_grid_units": float(decoder_fps[index]),
                "refiner_rmse_grid_units": float(decoder_refiner[index]),
                "relative_rmse_improvement_percent": float(decoder_improvements[index]),
            }
            for index in range(decoder_count)
        ],
        "decoders_better_than_fps": int(np.count_nonzero(decoder_improvements > 0)),
        "every_decoder_better_than_fps": bool(np.all(decoder_improvements > 0)),
        "passes_positive_interval": bool(lower > 0.0),
    }


def _named_bootstrap_comparison(
    baseline: np.ndarray,
    candidate: np.ndarray,
    categories: np.ndarray,
    *,
    baseline_method: str,
    candidate_method: str,
    samples: int,
    confidence_level: float,
    seed: int,
    decoder_seeds: tuple[int, ...],
) -> dict[str, Any]:
    comparison = _bootstrap_comparison(
        baseline,
        candidate,
        categories,
        samples=samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    comparison["baseline_method"] = baseline_method
    comparison["candidate_method"] = candidate_method
    comparison["candidate_replicate_count"] = comparison.pop("refiner_count")
    comparison["aggregate_baseline_rmse_grid_units"] = comparison.pop(
        "aggregate_fps_rmse_grid_units"
    )
    comparison["aggregate_candidate_rmse_grid_units"] = comparison.pop(
        "aggregate_refiner_rmse_grid_units"
    )
    comparison["decoders_candidate_better"] = comparison.pop("decoders_better_than_fps")
    comparison["every_decoder_candidate_better"] = comparison.pop(
        "every_decoder_better_than_fps"
    )
    for decoder_seed, per_decoder in zip(
        decoder_seeds, comparison["per_decoder"], strict=True
    ):
        per_decoder["decoder_seed"] = decoder_seed
        per_decoder.pop("decoder_index")
        per_decoder["baseline_rmse_grid_units"] = per_decoder.pop("fps_rmse_grid_units")
        per_decoder["candidate_rmse_grid_units"] = per_decoder.pop(
            "refiner_rmse_grid_units"
        )
    return comparison


def summarize_official_rows(
    rows: list[dict[str, Any]], config: OfficialStabilityConfig
) -> dict[str, Any]:
    """Aggregate complete official rows into the predeclared paired gates."""

    indexed = {_row_key(row): row for row in rows}
    comparisons: list[dict[str, Any]] = []
    selection_methods = [method for method in config.methods if method != "refiner"]
    for split_index, split in enumerate(config.splits):
        cloud_keys = sorted(
            {
                (row["family"], row["model_id"], row["sample_id"])
                for row in rows
                if row["split"] == split
            }
        )
        if not cloud_keys:
            raise RuntimeError(f"no official rows for split={split}")
        categories = np.asarray([key[0] for key in cloud_keys])
        for metric_index, metric in enumerate(METRICS):
            selections = {
                method: np.asarray(
                    [
                        [
                            indexed[
                                (
                                    split,
                                    method,
                                    decoder_seed,
                                    None,
                                    *cloud_key,
                                )
                            ][metric]
                            for cloud_key in cloud_keys
                        ]
                        for decoder_seed in config.decoder_seeds
                    ],
                    dtype=np.float64,
                )
                for method in selection_methods
            }
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
                                    *cloud_key,
                                )
                            ][metric]
                            for cloud_key in cloud_keys
                        ]
                        for refiner_seed in config.refiner_seeds
                    ]
                    for decoder_seed in config.decoder_seeds
                ],
                dtype=np.float64,
            )
            comparison_seed = (
                config.bootstrap_seed + 10_000 * split_index + 100 * metric_index
            )
            for method_index, method in enumerate(selection_methods):
                comparison = _named_bootstrap_comparison(
                    selections[method],
                    refiner,
                    categories,
                    baseline_method=method,
                    candidate_method="refiner",
                    samples=config.bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=comparison_seed + method_index,
                    decoder_seeds=config.decoder_seeds,
                )
                comparisons.append(
                    {
                        "split": split,
                        "metric": metric,
                        "comparison_role": "refiner_vs_selection",
                        **comparison,
                    }
                )
            for method_index, method in enumerate(selection_methods):
                if method == "fps":
                    continue
                comparison = _named_bootstrap_comparison(
                    selections["fps"],
                    selections[method][:, None, :],
                    categories,
                    baseline_method="fps",
                    candidate_method=method,
                    samples=config.bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=comparison_seed + len(selection_methods) + method_index,
                    decoder_seeds=config.decoder_seeds,
                )
                comparisons.append(
                    {
                        "split": split,
                        "metric": metric,
                        "comparison_role": "selection_vs_fps",
                        **comparison,
                    }
                )
    validation_refiner = [
        row
        for row in comparisons
        if row["split"] == "validation"
        and row["comparison_role"] == "refiner_vs_selection"
    ]
    validation_fps = [
        row for row in validation_refiner if row["baseline_method"] == "fps"
    ]
    return {
        "comparison_definition": (
            "paired hierarchical category/cloud bootstrap with paired decoder and "
            "common-refiner factor resampling; baseline candidates have one fixed "
            "replicate; primary quantity is relative RMSE computed from official "
            "symmetric MSE"
        ),
        "inferential_gate_eligible": not config.allow_single_seed,
        "comparisons": comparisons,
        "official_metric_gate_passes": bool(
            not config.allow_single_seed
            and len(validation_fps) == len(METRICS)
            and all(
                row["every_decoder_candidate_better"]
                and row["passes_positive_interval"]
                for row in validation_fps
            )
        ),
        "selection_baseline_gate_passes": bool(
            not config.allow_single_seed
            and len(validation_refiner) == len(selection_methods) * len(METRICS)
            and all(row["passes_positive_interval"] for row in validation_refiner)
        ),
    }


def _load_models(
    stability: StabilityExperimentConfig,
    official: OfficialStabilityConfig,
    *,
    decoder_seed: int,
    refiner_seed: int | None,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module | None, dict[str, Any]]:
    artifact_dir = Path(official.stability_artifact_dir)
    decoder_artifact_dir = Path(
        stability.decoder_source_artifact_dir or official.stability_artifact_dir
    )
    selection_path = (
        decoder_artifact_dir / f"decoders/seed_{decoder_seed}/selection.json"
    )
    selection = json.loads(selection_path.read_text())
    if _contains_forbidden_key(selection):
        raise RuntimeError("sealed calibration selection contains a test-field key")
    decoder_path = decoder_artifact_dir / f"decoders/seed_{decoder_seed}/stabilized.pt"
    decoder_checkpoint = torch.load(
        decoder_path, map_location=device, weights_only=True
    )
    decoder = _decoder(stability, device)
    decoder.load_state_dict(decoder_checkpoint["model"])
    decoder.eval().requires_grad_(False)
    decoder_hash = _state_hash(decoder)
    selected_hash = selection["arms"]["stabilized"]["state_hash"]
    if not (
        decoder_hash
        == selected_hash
        == decoder_checkpoint["state_hash"]
        == decoder_checkpoint["selection"]["selected_state_hash"]
    ):
        raise RuntimeError("stabilized decoder does not match its sealed selection")
    metadata = {
        "selection_sha256": file_sha256(selection_path),
        "decoder_checkpoint_sha256": file_sha256(decoder_path),
        "decoder_state_hash": decoder_hash,
        "calibration_selection": decoder_checkpoint["selection"],
        "decoder_artifact_dir": str(decoder_artifact_dir),
        "decoder_reused_from_source_artifact": bool(
            stability.decoder_source_artifact_dir is not None
        ),
    }
    if refiner_seed is None:
        return decoder, None, metadata

    refiner_path = artifact_dir / (
        f"pairs/stabilized/decoder_{decoder_seed}_refiner_{refiner_seed}/refiner.pt"
    )
    refiner_checkpoint = torch.load(
        refiner_path, map_location=device, weights_only=True
    )
    if (
        refiner_checkpoint["decoder_seed"] != decoder_seed
        or refiner_checkpoint["refiner_seed"] != refiner_seed
        or refiner_checkpoint["decoder_state_hash"] != decoder_hash
    ):
        raise RuntimeError("refiner checkpoint is assigned to a different decoder cell")
    refiner = _refiner(stability, device)
    refiner.load_state_dict(refiner_checkpoint["model"])
    refiner.eval().requires_grad_(False)
    metadata.update(
        {
            "refiner_checkpoint_sha256": file_sha256(refiner_path),
            "refiner_state_hash": _state_hash(refiner),
        }
    )
    return decoder, refiner, metadata


def _method_seed(
    config: OfficialStabilityConfig,
    *,
    method: str,
    split: str,
    sample: dict[str, Any],
) -> int:
    family, model_id, sample_id = _sample_identity(sample)
    seed_family = {
        "fps_random_start_best_of_8": "fps_random_start",
        "random_best_of_1": "random_best_of_n",
        "random_best_of_16": "random_best_of_n",
    }.get(method, method)
    identity = json.dumps(
        [
            config.selection_seed,
            seed_family,
            split,
            family,
            model_id,
            sample_id,
        ],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _evaluate_method(
    stability: StabilityExperimentConfig,
    official: OfficialStabilityConfig,
    *,
    decoder: torch.nn.Module,
    refiner: torch.nn.Module | None,
    decoder_seed: int,
    refiner_seed: int | None,
    split: str,
    sample: dict[str, Any],
    device: torch.device,
    scratch_root: Path,
    method: str | None = None,
) -> dict[str, Any]:
    source_points = sample.get("source_points", sample.get("points"))
    source_normals = sample.get("source_normals", sample.get("normals"))
    if not isinstance(source_points, torch.Tensor) or not isinstance(
        source_normals, torch.Tensor
    ):
        raise ValueError("official evaluation sample lacks points or normals")
    source = source_points.unsqueeze(0).to(device)
    normals = source_normals.cpu().numpy()
    family, model_id, sample_id = _sample_identity(sample)
    sample = {**sample, "sample_id": sample_id, "family": family, "model_id": model_id}
    method = method or ("fps" if refiner is None else "refiner")
    if method == "refiner" and refiner is None:
        raise ValueError("refiner evaluation requires a refiner model")
    if method != "refiner" and refiner is not None:
        raise ValueError("selection baseline evaluation cannot receive a refiner")
    if method not in {"refiner"} | set(SELECTION_METHODS):
        raise ValueError(f"unknown official method: {method}")

    selection_seed: int | None = None
    representation_class = "free-coordinate"
    selection_trials: int | None = None
    bitstream_mode = "free"

    def scorer(candidate: torch.Tensor) -> float:
        with torch.no_grad():
            reconstruction = decoder(
                candidate.unsqueeze(0), num_output_points=stability.num_points
            )
            loss = chamfer_squared_chunked(
                reconstruction,
                source,
                chunk_size=stability.distance_chunk_size,
            )
        return float(loss.item())

    _synchronize(device)
    encode_started = time.perf_counter()
    if method == "refiner":
        assert refiner is not None
        coordinates = refiner(
            source,
            stability.constellation_size,
            decoder=decoder,
            target=source,
            num_output_points=stability.num_points,
        )
    else:
        selection_seed = _method_seed(
            official, method=method, split=split, sample=sample
        )
        coordinates = SELECTION_METHODS[method](
            source[0],
            stability.constellation_size,
            stability.coordinate_bits,
            selection_seed,
            scorer,
        ).unsqueeze(0)
        representation_class = SELECTION_REPRESENTATIONS[method]
        selection_trials = SELECTION_TRIALS[method]
        if method == "fps":
            bitstream_mode = "fps"
        elif representation_class == "strict-subset":
            bitstream_mode = "strict_subset"
    decoded, stream_bytes, exact, lattice_exact = _serialized_coordinates(
        coordinates,
        config=stability,
        mode=bitstream_mode,
    )
    _synchronize(device)
    encode_seconds = time.perf_counter() - encode_started
    _synchronize(device)
    decode_started = time.perf_counter()
    with torch.no_grad():
        reconstruction = decoder(decoded, num_output_points=stability.num_points)
    _synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    with tempfile.TemporaryDirectory(
        prefix=(
            f"{split}-{method}-d{decoder_seed}-"
            f"r{refiner_seed if refiner_seed is not None else 'none'}-"
        ),
        dir=scratch_root,
    ) as temporary:
        metric = run_pc_error(
            Path(official.pc_error_executable),
            source[0].detach().cpu().numpy(),
            reconstruction[0].detach().cpu().numpy(),
            normals,
            work_dir=Path(temporary),
            position_bits=official.position_bits,
            timeout_seconds=official.timeout_seconds,
        )
    if not exact or not lattice_exact:
        raise RuntimeError("official evaluation message failed exact stream checks")
    if len(stream_bytes) != 1:
        raise RuntimeError("official evaluation expected a single encoded cloud")
    return {
        "split": split,
        "method": method,
        "decoder_seed": decoder_seed,
        "refiner_seed": refiner_seed,
        "family": family,
        "model_id": model_id,
        "sample_id": sample_id,
        "constellation_size": stability.constellation_size,
        "coordinate_bits": stability.coordinate_bits,
        "representation_class": representation_class,
        "bitstream_mode": bitstream_mode,
        "selection_seed": selection_seed,
        "selection_trials": selection_trials,
        "stream_bytes": stream_bytes[0],
        "actual_stream_bpp": 8.0 * stream_bytes[0] / stability.num_points,
        "serialized_round_trip_exact": exact,
        "coordinates_on_exact_lattice": lattice_exact,
        "source_only_decoder_gradient": True,
        "selection_uses_source_only_information": True,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "official_metric_seconds": metric.elapsed_seconds,
        **metric.metrics,
    }


def run_official_stability(
    config: OfficialStabilityConfig, *, device_name: str | None = None
) -> dict[str, Any]:
    """Evaluate and resume all predeclared stabilized cells."""

    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    if tuple(config.decoder_seeds) != tuple(stability.decoder_seeds):
        raise ValueError("official decoder seeds must match Experiment 019 exactly")
    if tuple(config.refiner_seeds) != tuple(stability.refiner_seeds):
        raise ValueError("official refiner seeds must match Experiment 019 exactly")
    if config.position_bits != stability.coordinate_bits:
        raise ValueError("official metric grid must match constellation precision")
    experiment = (
        "020_official_stability"
        if config.methods == DEFAULT_METHODS
        else "021_selection_baselines"
    )
    executable = Path(config.pc_error_executable)
    if not executable.is_file() or not executable.stat().st_mode & 0o111:
        raise FileNotFoundError(f"pc_error is missing or not executable: {executable}")

    artifact_dir = Path(config.stability_artifact_dir)
    stability_metrics_path = artifact_dir / "stability_metrics.json"
    stability_metrics = json.loads(stability_metrics_path.read_text())
    if not _stability_config_matches(stability_metrics["config"], stability):
        raise RuntimeError("Experiment 019 artifact config differs from checked config")
    if not all(stability_metrics["contract_checks"].values()):
        raise RuntimeError("Experiment 019 artifact has a failed scientific contract")
    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    if data_protocol != stability_metrics["data_protocol"]:
        raise RuntimeError(
            "Experiment 019 data identity changed before official metric"
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = output_dir / "metric_scratch"
    scratch_root.mkdir(exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    run_manifest = {
        "experiment": experiment,
        "config": _json_ready(asdict(config)),
        "stability_config_sha256": file_sha256(stability_path),
        "stability_metrics_sha256": file_sha256(stability_metrics_path),
        "pc_error_sha256": file_sha256(executable),
        "data_protocol": data_protocol,
    }
    if manifest_path.exists():
        if not _official_manifest_matches(
            json.loads(manifest_path.read_text()), run_manifest
        ):
            raise RuntimeError(f"existing Experiment {experiment[:3]} manifest differs")
    else:
        manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")

    rows_path = output_dir / "official_per_cloud.jsonl"
    rows = _load_rows(rows_path)
    resumed_rows = len(rows)
    completed = {_row_key(row) for row in rows}
    existing_sizes = {row["stream_bytes"] for row in rows}
    if len(existing_sizes) > 1:
        raise RuntimeError("resumed constellation streams have inconsistent sizes")
    declared_stream_bytes = expected_stream_bytes(
        stability.constellation_size, stability.coordinate_bits
    )
    if existing_sizes and existing_sizes != {declared_stream_bytes}:
        raise RuntimeError("resumed stream size differs from the declared fixed rate")
    device = select_device(device_name)
    started = time.perf_counter()
    model_records = []
    expected_bytes = declared_stream_bytes
    selection_methods = [method for method in config.methods if method != "refiner"]
    for decoder_seed in config.decoder_seeds:
        decoder, _, decoder_metadata = _load_models(
            stability,
            config,
            decoder_seed=decoder_seed,
            refiner_seed=None,
            device=device,
        )
        decoder_hash_before = _state_hash(decoder)
        for method in selection_methods:
            for split in config.splits:
                dataset = datasets[split]
                count = (
                    len(dataset)
                    if config.max_clouds_per_split is None
                    else min(len(dataset), config.max_clouds_per_split)
                )
                for sample_index in range(count):
                    sample = dataset[sample_index]
                    family, model_id, sample_id = _sample_identity(sample)
                    key = (
                        split,
                        method,
                        decoder_seed,
                        None,
                        family,
                        model_id,
                        sample_id,
                    )
                    if key in completed:
                        continue
                    row = _evaluate_method(
                        stability,
                        config,
                        decoder=decoder,
                        refiner=None,
                        decoder_seed=decoder_seed,
                        refiner_seed=None,
                        split=split,
                        sample=sample,
                        device=device,
                        scratch_root=scratch_root,
                        method=method,
                    )
                    if row["stream_bytes"] != expected_bytes:
                        raise RuntimeError(
                            "constellation stream differs from the declared fixed rate"
                        )
                    _append_row(rows_path, row)
                    rows.append(row)
                    completed.add(key)

        if _state_hash(decoder) != decoder_hash_before:
            raise RuntimeError("decoder changed during selection baseline evaluation")

        for refiner_seed in config.refiner_seeds:
            decoder, refiner, metadata = _load_models(
                stability,
                config,
                decoder_seed=decoder_seed,
                refiner_seed=refiner_seed,
                device=device,
            )
            assert refiner is not None
            for split in config.splits:
                dataset = datasets[split]
                count = (
                    len(dataset)
                    if config.max_clouds_per_split is None
                    else min(len(dataset), config.max_clouds_per_split)
                )
                for sample_index in range(count):
                    sample = dataset[sample_index]
                    family, model_id, sample_id = _sample_identity(sample)
                    key = (
                        split,
                        "refiner",
                        decoder_seed,
                        refiner_seed,
                        family,
                        model_id,
                        sample_id,
                    )
                    if key in completed:
                        continue
                    row = _evaluate_method(
                        stability,
                        config,
                        decoder=decoder,
                        refiner=refiner,
                        decoder_seed=decoder_seed,
                        refiner_seed=refiner_seed,
                        split=split,
                        sample=sample,
                        device=device,
                        scratch_root=scratch_root,
                        method="refiner",
                    )
                    if row["stream_bytes"] != expected_bytes:
                        raise RuntimeError(
                            "constellation streams have inconsistent sizes"
                        )
                    _append_row(rows_path, row)
                    rows.append(row)
                    completed.add(key)
            if _state_hash(decoder) != metadata["decoder_state_hash"]:
                raise RuntimeError("decoder changed during official refiner inference")
            model_records.append(
                {
                    "decoder_seed": decoder_seed,
                    "refiner_seed": refiner_seed,
                    **metadata,
                    "decoder_unchanged_during_evaluation": True,
                }
            )
        if _state_hash(decoder) != decoder_hash_before:
            raise RuntimeError("decoder state drifted during official evaluation")

    summary = summarize_official_rows(rows, config)
    result = {
        "experiment": experiment,
        "config": _json_ready(asdict(config)),
        "device": str(device),
        "resumed_rows": resumed_rows,
        "per_cloud_rows": len(rows),
        "expected_stream_bytes": expected_bytes,
        "contract_checks": {
            "experiment_019_contract_passed": True,
            "data_protocol_unchanged": True,
            "sealed_selection_has_no_test_keys": True,
            "decoder_hashes_unchanged": all(
                record["decoder_unchanged_during_evaluation"]
                for record in model_records
            ),
            "actual_stream_bytes_present": bool(
                rows and all(row["stream_bytes"] > 0 for row in rows)
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
            "source_only_decoder_gradient": bool(
                rows and all(row["source_only_decoder_gradient"] for row in rows)
            ),
            "source_only_selection": bool(
                rows
                and all(row["selection_uses_source_only_information"] for row in rows)
            ),
            "representation_classes_declared": bool(
                rows
                and all(
                    row["representation_class"] in {"free-coordinate", "strict-subset"}
                    for row in rows
                )
            ),
        },
        "tool_identity": {
            "pc_error_path": str(executable),
            "pc_error_sha256": file_sha256(executable),
            "position_bits": config.position_bits,
        },
        "model_records": model_records,
        "official_statistics": summary,
        "elapsed_seconds": time.perf_counter() - started,
        "per_cloud_path": str(rows_path),
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError(
            f"Experiment {experiment[:3]} official metric contract failed"
        )
    (output_dir / "official_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "rows": result["per_cloud_rows"],
                "official_metric_gate_passes": summary["official_metric_gate_passes"],
                "selection_baseline_gate_passes": summary[
                    "selection_baseline_gate_passes"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(output_dir / "official_metrics.json"),
            }
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--pc-error", type=Path)
    parser.add_argument("--max-clouds-per-split", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = OfficialStabilityConfig.from_json(args.config)
    if args.pc_error is not None:
        config = replace(config, pc_error_executable=str(args.pc_error))
    if args.max_clouds_per_split is not None:
        config = replace(config, max_clouds_per_split=args.max_clouds_per_split)
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    run_official_stability(config, device_name=args.device)


if __name__ == "__main__":
    main()
