"""Run fixed-data multi-seed external compression benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pointconstellation.refiner_benchmark import paired_hierarchical_bootstrap
from pointconstellation.standardized_benchmark import (
    StandardizedBenchmarkConfig,
    run_standardized_benchmark,
)


@dataclass(frozen=True)
class StandardizedMultiSeedConfig:
    """Configuration for paired external-data training seeds."""

    base_benchmark_config: str
    model_seeds: tuple[int, ...] = (7, 17, 29)
    data_seed: int = 1517
    primary_constellation_size: int = 8
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_814
    confidence_level: float = 0.95
    output_dir: str = "artifacts/local/experiment_017_modelnet40_multiseed"

    def __post_init__(self) -> None:
        if len(self.model_seeds) < 2 or len(set(self.model_seeds)) != len(
            self.model_seeds
        ):
            raise ValueError("model_seeds must contain at least two unique seeds")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

    @classmethod
    def from_json(cls, path: Path) -> StandardizedMultiSeedConfig:
        values = json.loads(path.read_text())
        if "model_seeds" in values:
            values["model_seeds"] = tuple(values["model_seeds"])
        return cls(**values)


def _row_index(
    rows: list[dict[str, Any]],
    *,
    split: str,
    method: str,
    constellation_size: int,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    selected = {}
    for row in rows:
        if (
            row["split"] != split
            or row["method"] != method
            or row["constellation_size"] != constellation_size
        ):
            continue
        key = (row["family"], row["model_id"], row["sample_id"])
        if key in selected:
            raise ValueError(f"duplicate per-cloud benchmark row: {key}")
        selected[key] = row
    if not selected:
        raise ValueError(
            f"no rows for split={split}, method={method}, K={constellation_size}"
        )
    return selected


def _paired_statistics(
    seed_results: list[dict[str, Any]],
    *,
    constellation_sizes: tuple[int, ...],
    samples: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Compute paired seed-and-cloud bootstrap statistics from exact rows."""

    splits = sorted({row["split"] for row in seed_results[0]["per_cloud"]})
    comparisons = []
    for split_index, split in enumerate(splits):
        for size_index, constellation_size in enumerate(constellation_sizes):
            fps_indices = [
                _row_index(
                    result["per_cloud"],
                    split=split,
                    method="fps",
                    constellation_size=constellation_size,
                )
                for result in seed_results
            ]
            keys = tuple(sorted(fps_indices[0]))
            if any(tuple(sorted(index)) != keys for index in fps_indices[1:]):
                raise RuntimeError("multi-seed FPS rows do not share cloud identities")
            for method_index, method in enumerate(("free", "strict_subset")):
                method_indices = [
                    _row_index(
                        result["per_cloud"],
                        split=split,
                        method=method,
                        constellation_size=constellation_size,
                    )
                    for result in seed_results
                ]
                if any(tuple(sorted(index)) != keys for index in method_indices):
                    raise RuntimeError(
                        "multi-seed method rows do not share cloud identities"
                    )
                metrics = {}
                for metric_index, field in enumerate(
                    ("chamfer_mse", "fresh_chamfer_mse")
                ):
                    baseline = np.asarray(
                        [[index[key][field] for key in keys] for index in fps_indices]
                    )
                    candidate = np.asarray(
                        [
                            [index[key][field] for key in keys]
                            for index in method_indices
                        ]
                    )
                    statistics = paired_hierarchical_bootstrap(
                        baseline,
                        candidate,
                        samples=samples,
                        confidence_level=confidence_level,
                        seed=(
                            seed
                            + 10_000 * split_index
                            + 1_000 * size_index
                            + 100 * method_index
                            + metric_index
                        ),
                    )
                    per_seed = []
                    for model_seed, baseline_row, candidate_row in zip(
                        (
                            result["config"]["experiment"]["seed"]
                            for result in seed_results
                        ),
                        baseline,
                        candidate,
                        strict=True,
                    ):
                        baseline_rmse = math.sqrt(float(baseline_row.mean()))
                        candidate_rmse = math.sqrt(float(candidate_row.mean()))
                        per_seed.append(
                            {
                                "model_seed": model_seed,
                                "relative_rmse_improvement_percent": 100.0
                                * (baseline_rmse - candidate_rmse)
                                / max(baseline_rmse, 1e-12),
                                "cloud_wins": int(
                                    np.count_nonzero(candidate_row < baseline_row)
                                ),
                                "clouds": len(keys),
                            }
                        )
                    metrics[field] = {**statistics, "per_seed": per_seed}
                comparisons.append(
                    {
                        "split": split,
                        "method": method,
                        "constellation_size": constellation_size,
                        "metrics": metrics,
                    }
                )
    return comparisons


def _per_family_statistics(
    seed_results: list[dict[str, Any]],
    *,
    constellation_sizes: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Return descriptive paired results for every represented category."""

    splits = sorted({row["split"] for row in seed_results[0]["per_cloud"]})
    comparisons = []
    for split in splits:
        for constellation_size in constellation_sizes:
            fps_indices = [
                _row_index(
                    result["per_cloud"],
                    split=split,
                    method="fps",
                    constellation_size=constellation_size,
                )
                for result in seed_results
            ]
            keys = tuple(sorted(fps_indices[0]))
            families = sorted({key[0] for key in keys})
            for method in ("free", "strict_subset"):
                method_indices = [
                    _row_index(
                        result["per_cloud"],
                        split=split,
                        method=method,
                        constellation_size=constellation_size,
                    )
                    for result in seed_results
                ]
                for family in families:
                    family_keys = tuple(key for key in keys if key[0] == family)
                    baseline = np.asarray(
                        [
                            [index[key]["chamfer_mse"] for key in family_keys]
                            for index in fps_indices
                        ]
                    )
                    candidate = np.asarray(
                        [
                            [index[key]["chamfer_mse"] for key in family_keys]
                            for index in method_indices
                        ]
                    )
                    baseline_rmse = math.sqrt(float(baseline.mean()))
                    candidate_rmse = math.sqrt(float(candidate.mean()))
                    comparisons.append(
                        {
                            "split": split,
                            "family": family,
                            "method": method,
                            "constellation_size": constellation_size,
                            "seeds": len(seed_results),
                            "clouds_per_seed": len(family_keys),
                            "aggregate_fps_rmse": baseline_rmse,
                            "aggregate_method_rmse": candidate_rmse,
                            "relative_rmse_improvement_percent": 100.0
                            * (baseline_rmse - candidate_rmse)
                            / max(baseline_rmse, 1e-12),
                            "seed_cloud_wins": int(
                                np.count_nonzero(candidate < baseline)
                            ),
                            "seed_cloud_comparisons": int(candidate.size),
                        }
                    )
    return comparisons


def _representative_examples(
    seed_results: list[dict[str, Any]],
    *,
    constellation_size: int,
    examples_per_tail: int = 3,
) -> list[dict[str, Any]]:
    """Record the strongest improvements and worst cases without cherry-picking."""

    splits = sorted({row["split"] for row in seed_results[0]["per_cloud"]})
    examples = []
    for model_seed, result in zip(
        (result["config"]["experiment"]["seed"] for result in seed_results),
        seed_results,
        strict=True,
    ):
        for split in splits:
            baseline = _row_index(
                result["per_cloud"],
                split=split,
                method="fps",
                constellation_size=constellation_size,
            )
            candidate = _row_index(
                result["per_cloud"],
                split=split,
                method="free",
                constellation_size=constellation_size,
            )
            ranked = sorted(
                (
                    {
                        "model_seed": model_seed,
                        "split": split,
                        "family": key[0],
                        "model_id": key[1],
                        "sample_id": key[2],
                        "fps_chamfer_rmse": math.sqrt(baseline[key]["chamfer_mse"]),
                        "free_chamfer_rmse": math.sqrt(candidate[key]["chamfer_mse"]),
                        "free_minus_fps_chamfer_mse": candidate[key]["chamfer_mse"]
                        - baseline[key]["chamfer_mse"],
                        "free_is_worse": candidate[key]["chamfer_mse"]
                        > baseline[key]["chamfer_mse"],
                    }
                    for key in baseline
                ),
                key=lambda row: row["free_minus_fps_chamfer_mse"],
            )
            for tail, selected in (
                ("largest_improvement", ranked[:examples_per_tail]),
                ("worst_case", ranked[-examples_per_tail:][::-1]),
            ):
                examples.extend({**row, "selection": tail} for row in selected)
    return examples


def run_standardized_multiseed(
    config: StandardizedMultiSeedConfig,
    *,
    device_name: str = "auto",
    resume: bool = False,
    aggregate_only: bool = False,
) -> dict[str, Any]:
    """Run independent learned seeds and one shared conventional-codec arm."""

    base = StandardizedBenchmarkConfig.from_json(Path(config.base_benchmark_config))
    if config.primary_constellation_size not in base.experiment.constellation_sizes:
        raise ValueError("primary constellation size is absent from base config")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "multiseed_metrics.json"
    previous_elapsed_seconds = None
    if aggregate_only and metrics_path.exists():
        previous_elapsed_seconds = json.loads(metrics_path.read_text()).get(
            "elapsed_seconds"
        )
    seed_results = []
    started = time.perf_counter()
    for seed_index, model_seed in enumerate(config.model_seeds):
        seed_dir = output_dir / f"seed_{model_seed}"
        experiment = replace(
            base.experiment,
            seed=model_seed,
            data_seed=config.data_seed,
            output_dir=str(seed_dir / "model"),
        )
        seed_config = replace(
            base,
            experiment=experiment,
            output_dir=str(seed_dir),
            gpcc=base.gpcc if seed_index == 0 else None,
        )
        seed_metrics_path = seed_dir / "benchmark_metrics.json"
        if aggregate_only:
            if not seed_metrics_path.exists():
                raise FileNotFoundError(
                    f"missing seed metrics for aggregate-only run: {seed_metrics_path}"
                )
            seed_result = json.loads(seed_metrics_path.read_text())
            seed_experiment = seed_result.get("config", {}).get("experiment", {})
            if (
                seed_experiment.get("seed") != model_seed
                or seed_experiment.get("data_seed") != config.data_seed
            ):
                raise RuntimeError("existing seed metrics do not match config")
            seed_results.append(seed_result)
        else:
            seed_results.append(
                run_standardized_benchmark(
                    seed_config,
                    device_name=device_name,
                    resume=resume,
                )
            )

    identities = [result["protocol"]["data_identity"] for result in seed_results]
    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("multi-seed runs do not share a data identity")
    decoder_hashes = [result["model"]["decoder_state_hash"] for result in seed_results]
    refiner_hashes = [result["model"]["refiner_state_hash"] for result in seed_results]
    if len(set(decoder_hashes)) != len(decoder_hashes):
        raise RuntimeError("independent seeds produced duplicate decoder hashes")
    if None in refiner_hashes or len(set(refiner_hashes)) != len(refiner_hashes):
        raise RuntimeError("independent seeds produced invalid refiner hashes")

    comparisons = _paired_statistics(
        seed_results,
        constellation_sizes=base.experiment.constellation_sizes,
        samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
        seed=config.bootstrap_seed,
    )
    family_comparisons = _per_family_statistics(
        seed_results,
        constellation_sizes=base.experiment.constellation_sizes,
    )
    representative_examples = _representative_examples(
        seed_results,
        constellation_size=config.primary_constellation_size,
    )
    primary = [
        row
        for row in comparisons
        if row["method"] == "free"
        and row["constellation_size"] == config.primary_constellation_size
    ]
    primary_passes = bool(primary) and all(
        row["metrics"]["chamfer_mse"]["confidence_interval_lower_percent"] > 0
        and all(
            seed_row["relative_rmse_improvement_percent"] > 0
            for seed_row in row["metrics"]["chamfer_mse"]["per_seed"]
        )
        for row in primary
    )
    result = {
        "config": asdict(config),
        "protocol": seed_results[0]["protocol"],
        "data_identity": identities[0],
        "model_independence": {
            "decoder_hashes": decoder_hashes,
            "refiner_hashes": refiner_hashes,
            "all_unique": True,
        },
        "per_seed": [
            {
                "model_seed": model_seed,
                "metrics_path": str(
                    output_dir / f"seed_{model_seed}" / "benchmark_metrics.json"
                ),
                "model": seed_result["model"],
                "summary": seed_result["summary"],
                "evaluation_elapsed_seconds": seed_result["evaluation_elapsed_seconds"],
            }
            for model_seed, seed_result in zip(
                config.model_seeds, seed_results, strict=True
            )
        ],
        "paired_comparisons": comparisons,
        "per_family_comparisons": family_comparisons,
        "representative_examples": representative_examples,
        "primary_gate": {
            "constellation_size": config.primary_constellation_size,
            "method": "free",
            "requires_every_seed_positive_and_ci_lower_above_zero": True,
            "passes": primary_passes,
        },
        "gpcc": seed_results[0]["gpcc"],
        "interpolation_free_gpcc_comparisons": seed_results[0][
            "interpolation_free_gpcc_comparisons"
        ],
        "elapsed_seconds": (
            previous_elapsed_seconds
            if previous_elapsed_seconds is not None
            else time.perf_counter() - started
        ),
        "aggregation_elapsed_seconds": time.perf_counter() - started,
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    config = StandardizedMultiSeedConfig.from_json(args.config)
    result = run_standardized_multiseed(
        config,
        device_name=args.device,
        resume=args.resume,
        aggregate_only=args.aggregate_only,
    )
    print(
        json.dumps(
            {
                "primary_gate": result["primary_gate"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "multiseed_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
