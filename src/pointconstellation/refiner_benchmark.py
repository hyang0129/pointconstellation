"""Run a multi-seed, paired statistical benchmark for Experiment 005."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pointconstellation.data import ProceduralPointCloudDataset, generate_sample
from pointconstellation.losses import pairwise_squared
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.refiner import CompetitiveConstellationRefiner
from pointconstellation.refiner_experiment import (
    RefinerExperimentConfig,
    _fps,
    _state_hash,
    run_refiner_experiment,
)
from pointconstellation.train import select_device

METHODS = (
    "fps",
    "no_feedback_free_coordinates",
    "no_feedback_strict_projection",
    "input_gradient_free_coordinates",
    "input_gradient_strict_projection",
)


@dataclass(frozen=True)
class RefinerBenchmarkConfig:
    """Configuration for a matched multi-seed Experiment 005 benchmark."""

    base_experiment_config: str = "configs/experiment_005_refiner_scale.json"
    model_seeds: tuple[int, ...] = (7, 17, 29)
    data_seed: int = 7
    primary_input_size: int = 256
    primary_constellation_size: int = 16
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_813
    confidence_level: float = 0.95
    output_dir: str = "artifacts/local/experiment_013_refiner_multiseed"

    def __post_init__(self) -> None:
        if len(self.model_seeds) < 2 or len(set(self.model_seeds)) != len(
            self.model_seeds
        ):
            raise ValueError("model_seeds must contain at least two unique seeds")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")
        if self.primary_input_size < 8 or self.primary_constellation_size < 2:
            raise ValueError("primary input and constellation sizes are invalid")

    @classmethod
    def from_json(cls, path: Path) -> RefinerBenchmarkConfig:
        values = json.loads(path.read_text())
        if "model_seeds" in values:
            values["model_seeds"] = tuple(values["model_seeds"])
        return cls(**values)


def paired_hierarchical_bootstrap(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap a paired aggregate RMSE improvement over seeds and clouds.

    Rows are independently trained model seeds and columns are the same held-out
    clouds for every seed. Each replicate resamples both axes with replacement,
    preserving the pairing between the baseline and method losses.
    """

    baseline = np.asarray(baseline, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)
    if baseline.ndim != 2 or method.shape != baseline.shape:
        raise ValueError("baseline and method must have the same 2D shape")
    if baseline.shape[0] < 2 or baseline.shape[1] < 2:
        raise ValueError("bootstrap requires at least two seeds and two clouds")
    if np.any(baseline < 0) or np.any(method < 0):
        raise ValueError("squared Chamfer losses cannot be negative")

    def improvement(first: np.ndarray, second: np.ndarray) -> float:
        baseline_rmse = math.sqrt(float(first.mean()))
        method_rmse = math.sqrt(float(second.mean()))
        return 100.0 * (baseline_rmse - method_rmse) / max(baseline_rmse, 1e-12)

    rng = np.random.default_rng(seed)
    seed_count, cloud_count = baseline.shape
    seed_indices = rng.integers(0, seed_count, size=(samples, seed_count))
    cloud_indices = rng.integers(0, cloud_count, size=(samples, cloud_count))
    first_draws = baseline[seed_indices[:, :, None], cloud_indices[:, None, :]].mean(
        axis=(1, 2)
    )
    second_draws = method[seed_indices[:, :, None], cloud_indices[:, None, :]].mean(
        axis=(1, 2)
    )
    bootstrap_improvement = (
        100.0
        * (np.sqrt(first_draws) - np.sqrt(second_draws))
        / np.sqrt(first_draws).clip(min=1e-12)
    )
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_improvement, (alpha, 1.0 - alpha))
    point = improvement(baseline, method)
    return {
        "seed_count": seed_count,
        "cloud_count": cloud_count,
        "bootstrap_samples": samples,
        "confidence_level": confidence_level,
        "aggregate_baseline_rmse": math.sqrt(float(baseline.mean())),
        "aggregate_method_rmse": math.sqrt(float(method.mean())),
        "relative_rmse_improvement_percent": point,
        "confidence_interval_lower_percent": float(lower),
        "confidence_interval_upper_percent": float(upper),
    }


def _dataset_manifest(
    *, data_seed: int, num_points: int, split: str, count: int
) -> dict[str, Any]:
    digest = hashlib.sha256()
    families: dict[str, int] = {}
    for sample_id in range(count):
        sample = generate_sample(
            sample_id,
            num_points=num_points,
            seed=data_seed,
            split=split,
        )
        digest.update(str(sample.sample_id).encode())
        digest.update(sample.family.encode())
        digest.update(sample.points.tobytes())
        digest.update(sample.normals.tobytes())
        families[sample.family] = families.get(sample.family, 0) + 1
    return {
        "split": split,
        "count": count,
        "num_points": num_points,
        "data_seed": data_seed,
        "sha256": digest.hexdigest(),
        "family_counts": dict(sorted(families.items())),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _completed_arm_matches(
    metrics_path: Path,
    config: RefinerExperimentConfig,
) -> bool:
    if not metrics_path.exists():
        return False
    metrics = _load_json(metrics_path)
    expected_config = json.loads(json.dumps(asdict(config)))
    return (
        metrics.get("config") == expected_config
        and metrics.get("decoder_unchanged") is True
    )


def _run_arm(
    config: RefinerExperimentConfig,
    *,
    device_name: str,
    force: bool,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    metrics_path = output_dir / "metrics.json"
    if not force and _completed_arm_matches(metrics_path, config):
        print(
            json.dumps(
                {
                    "stage": "reuse_completed_arm",
                    "seed": config.seed,
                    "use_decoder_gradient": config.use_decoder_gradient,
                    "metrics": str(metrics_path),
                }
            )
        )
        return _load_json(metrics_path)
    return run_refiner_experiment(config, device_name=device_name)


def _load_models(
    *,
    decoder_path: Path,
    feedback_refiner_path: Path,
    no_feedback_refiner_path: Path,
    config: RefinerExperimentConfig,
    device: torch.device,
) -> tuple[
    VariableConstellationDecoder,
    CompetitiveConstellationRefiner,
    CompetitiveConstellationRefiner,
]:
    decoder = VariableConstellationDecoder(
        config.num_points,
        max(config.constellation_sizes),
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    decoder_checkpoint = torch.load(
        decoder_path, map_location=device, weights_only=True
    )
    decoder.load_state_dict(decoder_checkpoint["model"])
    decoder.eval().requires_grad_(False)

    def load_refiner(path: Path, *, use_decoder_gradient: bool):
        refiner = CompetitiveConstellationRefiner(
            max(config.constellation_sizes),
            bits=config.bits,
            feature_width=config.feature_width,
            num_heads=config.num_heads,
            recurrent_steps=config.recurrent_steps,
            responsibility_temperature=config.responsibility_temperature,
            maximum_update=config.maximum_update,
            use_decoder_gradient=use_decoder_gradient,
        ).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        refiner.load_state_dict(checkpoint["model"])
        return refiner.eval().requires_grad_(False)

    feedback = load_refiner(feedback_refiner_path, use_decoder_gradient=True)
    no_feedback = load_refiner(no_feedback_refiner_path, use_decoder_gradient=False)
    return decoder, feedback, no_feedback


def _chamfer_per_cloud(reconstruction: Tensor, target: Tensor) -> Tensor:
    distances = pairwise_squared(reconstruction, target)
    return 0.5 * (distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1))


def _evaluate_primary(
    *,
    decoder: VariableConstellationDecoder,
    feedback: CompetitiveConstellationRefiner,
    no_feedback: CompetitiveConstellationRefiner,
    experiment: RefinerExperimentConfig,
    benchmark: RefinerBenchmarkConfig,
    split: str,
    count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    dataset = ProceduralPointCloudDataset(
        count,
        num_points=experiment.num_points,
        seed=benchmark.data_seed,
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=experiment.batch_size,
        shuffle=False,
        num_workers=0,
    )
    records: list[dict[str, Any]] = []
    for batch in loader:
        target = batch["points"].to(device)
        source = target[:, : benchmark.primary_input_size]
        with torch.no_grad():
            fps = _fps(
                source,
                benchmark.primary_constellation_size,
                experiment.bits,
            )
            feedback_free = feedback(
                source,
                benchmark.primary_constellation_size,
                decoder=decoder,
                target=source,
                num_output_points=experiment.num_points,
            )
            no_feedback_free = no_feedback(
                source,
                benchmark.primary_constellation_size,
                decoder=decoder,
                target=source,
                num_output_points=experiment.num_points,
            )
            constellations = {
                "fps": fps,
                "no_feedback_free_coordinates": no_feedback_free,
                "no_feedback_strict_projection": (
                    no_feedback.project_unique_to_input(no_feedback_free, source)
                ),
                "input_gradient_free_coordinates": feedback_free,
                "input_gradient_strict_projection": (
                    feedback.project_unique_to_input(feedback_free, source)
                ),
            }
            losses = {
                method: _chamfer_per_cloud(
                    decoder(
                        constellation,
                        num_output_points=experiment.num_points,
                    ),
                    target,
                )
                .detach()
                .cpu()
                for method, constellation in constellations.items()
            }
        for batch_index, sample_id in enumerate(batch["sample_id"].tolist()):
            records.append(
                {
                    "sample_id": int(sample_id),
                    "family": batch["family"][batch_index],
                    "squared_chamfer": {
                        method: float(values[batch_index].item())
                        for method, values in losses.items()
                    },
                }
            )
    return records


def _matrix_for_method(
    seed_results: list[dict[str, Any]], method: str
) -> tuple[np.ndarray, list[int], list[str]]:
    first_records = seed_results[0]["per_cloud"]
    sample_ids = [record["sample_id"] for record in first_records]
    families = [record["family"] for record in first_records]
    rows = []
    for seed_result in seed_results:
        records = seed_result["per_cloud"]
        if [record["sample_id"] for record in records] != sample_ids:
            raise RuntimeError("held-out sample IDs differ across model seeds")
        if [record["family"] for record in records] != families:
            raise RuntimeError("held-out families differ across model seeds")
        rows.append([record["squared_chamfer"][method] for record in records])
    return np.asarray(rows), sample_ids, families


def _aggregate_rmse(values: np.ndarray) -> float:
    return math.sqrt(float(np.asarray(values).mean()))


def _distribution_summary(baseline: np.ndarray, values: np.ndarray) -> dict[str, float]:
    baseline_cloud_rmse = np.sqrt(np.asarray(baseline, dtype=np.float64))
    method_cloud_rmse = np.sqrt(np.asarray(values, dtype=np.float64))
    paired_improvement = (
        100.0
        * (baseline_cloud_rmse - method_cloud_rmse)
        / baseline_cloud_rmse.clip(min=1e-12)
    )
    return {
        "median_per_cloud_rmse": float(np.median(method_cloud_rmse)),
        "p95_per_cloud_rmse": float(np.quantile(method_cloud_rmse, 0.95)),
        "median_paired_relative_improvement_percent": float(
            np.median(paired_improvement)
        ),
        "p10_paired_relative_improvement_percent": float(
            np.quantile(paired_improvement, 0.10)
        ),
    }


def _summarize_split(
    seed_results: list[dict[str, Any]],
    *,
    benchmark: RefinerBenchmarkConfig,
    split_index: int,
) -> dict[str, Any]:
    baseline, sample_ids, families = _matrix_for_method(seed_results, "fps")
    method_matrices = {
        method: _matrix_for_method(seed_results, method)[0] for method in METHODS
    }
    methods: dict[str, Any] = {}
    for method_index, (method, values) in enumerate(method_matrices.items()):
        bootstrap = paired_hierarchical_bootstrap(
            baseline,
            values,
            samples=benchmark.bootstrap_samples,
            confidence_level=benchmark.confidence_level,
            seed=(benchmark.bootstrap_seed + 1000 * split_index + method_index),
        )
        methods[method] = {
            **bootstrap,
            **_distribution_summary(baseline, values),
            "per_seed": [
                {
                    "model_seed": seed_results[index]["model_seed"],
                    "rmse": _aggregate_rmse(values[index]),
                    "relative_rmse_improvement_percent": 100.0
                    * (
                        _aggregate_rmse(baseline[index])
                        - _aggregate_rmse(values[index])
                    )
                    / max(_aggregate_rmse(baseline[index]), 1e-12),
                }
                for index in range(len(seed_results))
            ],
        }

    mechanism_contrast = paired_hierarchical_bootstrap(
        method_matrices["no_feedback_free_coordinates"],
        method_matrices["input_gradient_free_coordinates"],
        samples=benchmark.bootstrap_samples,
        confidence_level=benchmark.confidence_level,
        seed=benchmark.bootstrap_seed + 1000 * split_index + len(METHODS),
    )

    family_summaries: dict[str, Any] = {}
    family_array = np.asarray(families)
    for family in sorted(set(families)):
        selected = family_array == family
        family_summaries[family] = {
            method: {
                "rmse": _aggregate_rmse(values[:, selected]),
                "relative_rmse_improvement_percent": 100.0
                * (
                    _aggregate_rmse(baseline[:, selected])
                    - _aggregate_rmse(values[:, selected])
                )
                / max(_aggregate_rmse(baseline[:, selected]), 1e-12),
            }
            for method, values in method_matrices.items()
        }

    return {
        "sample_ids": sample_ids,
        "methods": methods,
        "input_gradient_vs_no_feedback": mechanism_contrast,
        "by_family": family_summaries,
    }


def _primary_curve(
    metrics: dict[str, Any],
    *,
    split: str,
    input_size: int,
    constellation_size: int,
) -> list[dict[str, Any]]:
    matches = [
        row["curve"]
        for row in metrics["evaluation"][split]
        if row["input_size"] == input_size
        and row["constellation_size"] == constellation_size
    ]
    if len(matches) != 1:
        raise RuntimeError("primary operating-point curve is missing or duplicated")
    return matches[0]


def _summarize_convergence(
    arm_metrics: list[dict[str, Any]],
    *,
    split: str,
    input_size: int,
    constellation_size: int,
) -> list[dict[str, Any]]:
    curves = [
        _primary_curve(
            metrics,
            split=split,
            input_size=input_size,
            constellation_size=constellation_size,
        )
        for metrics in arm_metrics
    ]
    if len({len(curve) for curve in curves}) != 1:
        raise RuntimeError("refinement curves have different lengths across seeds")
    summary = []
    for step_index in range(len(curves[0])):
        modes = {}
        for mode in ("free", "strict_subset"):
            per_seed = [curve[step_index][mode]["chamfer_rmse"] for curve in curves]
            modes[mode] = {
                "aggregate_chamfer_rmse": math.sqrt(
                    sum(value * value for value in per_seed) / len(per_seed)
                ),
                "per_seed_chamfer_rmse": per_seed,
            }
        summary.append({"step": step_index, **modes})
    return summary


def run_refiner_benchmark(
    config: RefinerBenchmarkConfig,
    *,
    device_name: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Train matched arms across seeds and compute paired primary statistics."""

    started = time.perf_counter()
    device = select_device(device_name)
    base = RefinerExperimentConfig.from_json(Path(config.base_experiment_config))
    if config.primary_input_size not in base.input_sizes:
        raise ValueError("primary_input_size is absent from the base experiment")
    if config.primary_constellation_size not in base.constellation_sizes:
        raise ValueError(
            "primary_constellation_size is absent from the base experiment"
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {
        "validation": base.validation_samples,
        "parameter_ood": base.parameter_ood_samples,
    }
    manifests = {
        split: _dataset_manifest(
            data_seed=config.data_seed,
            num_points=base.num_points,
            split=split,
            count=count,
        )
        for split, count in {
            "train": base.train_samples,
            **split_counts,
        }.items()
    }

    seed_runs: list[dict[str, Any]] = []
    split_seed_results: dict[str, list[dict[str, Any]]] = {
        split: [] for split in split_counts
    }
    feedback_arm_metrics: list[dict[str, Any]] = []
    no_feedback_arm_metrics: list[dict[str, Any]] = []
    for model_seed in config.model_seeds:
        seed_dir = output_dir / f"seed_{model_seed}"
        feedback_dir = seed_dir / "input_gradient"
        no_feedback_dir = seed_dir / "no_feedback"
        feedback_config = replace(
            base,
            seed=model_seed,
            data_seed=config.data_seed,
            use_decoder_gradient=True,
            decoder_checkpoint=None,
            output_dir=str(feedback_dir),
        )
        print(json.dumps({"stage": "seed_start", "model_seed": model_seed}))
        feedback_metrics = _run_arm(
            feedback_config,
            device_name=device_name,
            force=force,
        )
        decoder_path = feedback_dir / "decoder.pt"
        no_feedback_config = replace(
            base,
            seed=model_seed,
            data_seed=config.data_seed,
            use_decoder_gradient=False,
            decoder_checkpoint=str(decoder_path),
            output_dir=str(no_feedback_dir),
        )
        no_feedback_metrics = _run_arm(
            no_feedback_config,
            device_name=device_name,
            force=force,
        )
        feedback_arm_metrics.append(feedback_metrics)
        no_feedback_arm_metrics.append(no_feedback_metrics)
        matched_initialization = (
            feedback_metrics["refiner_hash_before_training"]
            == no_feedback_metrics["refiner_hash_before_training"]
        )
        matched_training_order = (
            feedback_metrics["refiner_training_order_hash"]
            == no_feedback_metrics["refiner_training_order_hash"]
        )
        matched_decoder = (
            feedback_metrics["decoder_hash_before_refiner"]
            == no_feedback_metrics["decoder_hash_before_refiner"]
        )
        if not (matched_initialization and matched_training_order and matched_decoder):
            raise RuntimeError(f"seed {model_seed} arms are not properly matched")

        decoder, feedback, no_feedback = _load_models(
            decoder_path=decoder_path,
            feedback_refiner_path=feedback_dir / "refiner.pt",
            no_feedback_refiner_path=no_feedback_dir / "refiner.pt",
            config=feedback_config,
            device=device,
        )
        decoder_hash = _state_hash(decoder)
        split_artifacts = {}
        for split, count in split_counts.items():
            per_cloud = _evaluate_primary(
                decoder=decoder,
                feedback=feedback,
                no_feedback=no_feedback,
                experiment=feedback_config,
                benchmark=config,
                split=split,
                count=count,
                device=device,
            )
            split_result = {
                "model_seed": model_seed,
                "per_cloud": per_cloud,
            }
            split_seed_results[split].append(split_result)
            split_path = seed_dir / f"primary_{split}.json"
            split_path.write_text(json.dumps(split_result, indent=2) + "\n")
            split_artifacts[split] = str(split_path)
        seed_runs.append(
            {
                "model_seed": model_seed,
                "decoder_hash": decoder_hash,
                "decoder_unchanged_in_both_arms": (
                    feedback_metrics["decoder_unchanged"]
                    and no_feedback_metrics["decoder_unchanged"]
                ),
                "matched_refiner_initialization": matched_initialization,
                "matched_refiner_training_order": matched_training_order,
                "matched_decoder": matched_decoder,
                "input_gradient_metrics": str(feedback_dir / "metrics.json"),
                "no_feedback_metrics": str(no_feedback_dir / "metrics.json"),
                "primary_split_artifacts": split_artifacts,
            }
        )

    decoder_hashes = [run["decoder_hash"] for run in seed_runs]
    independently_trained_decoders = len(set(decoder_hashes)) == len(decoder_hashes)
    split_summaries = {
        split: _summarize_split(
            seed_results,
            benchmark=config,
            split_index=split_index,
        )
        for split_index, (split, seed_results) in enumerate(split_seed_results.items())
    }
    convergence = {
        split: {
            "input_gradient": _summarize_convergence(
                feedback_arm_metrics,
                split=split,
                input_size=config.primary_input_size,
                constellation_size=config.primary_constellation_size,
            ),
            "no_feedback": _summarize_convergence(
                no_feedback_arm_metrics,
                split=split,
                input_size=config.primary_input_size,
                constellation_size=config.primary_constellation_size,
            ),
        }
        for split in split_counts
    }
    primary_method = "input_gradient_free_coordinates"
    all_seed_wins = all(
        seed["relative_rmse_improvement_percent"] > 0
        for split in split_summaries.values()
        for seed in split["methods"][primary_method]["per_seed"]
    )
    confidence_excludes_zero = all(
        split["methods"][primary_method]["confidence_interval_lower_percent"] > 0
        for split in split_summaries.values()
    )
    no_feedback_improves_any_split = any(
        split["methods"]["no_feedback_free_coordinates"][
            "relative_rmse_improvement_percent"
        ]
        > 0
        for split in split_summaries.values()
    )
    result = {
        "experiment": "013_refiner_multiseed_benchmark",
        "benchmark_config": asdict(config),
        "base_experiment_config": asdict(base),
        "device": str(device),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "primary_operating_point": {
            "input_size": config.primary_input_size,
            "constellation_size": config.primary_constellation_size,
            "bits_per_coordinate": base.bits,
            "nominal_coordinate_payload_bits": (
                3 * config.primary_constellation_size * base.bits
            ),
        },
        "dataset_manifests": manifests,
        "seed_runs": seed_runs,
        "independently_trained_decoder_hashes": independently_trained_decoders,
        "splits": split_summaries,
        "primary_convergence": convergence,
        "predeclared_gate": {
            "all_input_gradient_seed_runs_beat_fps": all_seed_wins,
            "input_gradient_bootstrap_ci_excludes_zero_on_both_splits": (
                confidence_excludes_zero
            ),
            "no_feedback_improves_at_least_one_split": (no_feedback_improves_any_split),
            "passed": (
                all_seed_wins
                and confidence_excludes_zero
                and no_feedback_improves_any_split
                and independently_trained_decoders
            ),
        },
        "message_contract": {
            "coordinate_only": True,
            "exact_quantization": True,
            "encoder_gradient_target_is_input_only": True,
            "strict_mode_is_post_hoc_projection": True,
            "actual_bitstream_rate_claim": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    metrics_path = output_dir / "benchmark_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_013_refiner_multiseed.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = RefinerBenchmarkConfig.from_json(args.config)
    result = run_refiner_benchmark(
        config,
        device_name=args.device,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "device": result["device"],
                "passed": result["predeclared_gate"]["passed"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "benchmark_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
