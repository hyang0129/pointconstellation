"""Run the Experiment 002 selected-rate architectural control."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from pointconstellation.train import MODEL_KINDS, TrainingConfig, train


@dataclass(frozen=True)
class SelectedRateSpec:
    base_config: TrainingConfig
    constellation_sizes: tuple[int, ...]
    model_kinds: tuple[str, ...]
    gated_model_kind: str = "relation"
    min_endpoint_improvement_percent: float = 1.0
    max_adjacent_regression_percent: float = 0.5

    def __post_init__(self) -> None:
        if len(self.constellation_sizes) < 2:
            raise ValueError("at least two constellation sizes are required")
        if tuple(sorted(set(self.constellation_sizes))) != self.constellation_sizes:
            raise ValueError("constellation_sizes must be unique and increasing")
        if any(
            size < 2 or size > self.base_config.num_points
            for size in self.constellation_sizes
        ):
            raise ValueError("constellation sizes must be between 2 and num_points")
        if not self.model_kinds or any(
            model_kind not in MODEL_KINDS for model_kind in self.model_kinds
        ):
            raise ValueError(f"model_kinds must be selected from {MODEL_KINDS}")
        if self.gated_model_kind not in self.model_kinds:
            raise ValueError("gated_model_kind must be included in model_kinds")
        if self.base_config.parameter_ood_samples < 1:
            raise ValueError("parameter_ood_samples must be positive")
        if self.min_endpoint_improvement_percent < 0:
            raise ValueError("min_endpoint_improvement_percent cannot be negative")
        if self.max_adjacent_regression_percent < 0:
            raise ValueError("max_adjacent_regression_percent cannot be negative")

    @classmethod
    def from_json(cls, path: Path) -> SelectedRateSpec:
        values = json.loads(path.read_text())
        return cls(
            base_config=TrainingConfig(**values["base_config"]),
            constellation_sizes=tuple(values["constellation_sizes"]),
            model_kinds=tuple(values["model_kinds"]),
            gated_model_kind=values.get("gated_model_kind", "relation"),
            min_endpoint_improvement_percent=values.get(
                "min_endpoint_improvement_percent", 1.0
            ),
            max_adjacent_regression_percent=values.get(
                "max_adjacent_regression_percent", 0.5
            ),
        )


def rate_curve_gate(
    runs: list[dict[str, Any]],
    *,
    model_kind: str,
    split: str,
    min_endpoint_improvement_percent: float,
    max_adjacent_regression_percent: float,
) -> dict[str, Any]:
    """Evaluate whether a model converts increasing coordinate rate to fidelity."""

    curve = sorted(
        (run for run in runs if run["model_kind"] == model_kind),
        key=lambda run: run["coordinate_payload_bits"],
    )
    if len(curve) < 2:
        raise ValueError("a rate gate requires at least two matching runs")
    if split not in {"validation", "parameter_ood"}:
        raise ValueError("split must be validation or parameter_ood")

    values = [run[split]["chamfer_rmse"] for run in curve]
    endpoint_improvement = 100.0 * (values[0] - values[-1]) / values[0]
    adjacent_regressions = [
        100.0 * (current - previous) / previous
        for previous, current in zip(values[:-1], values[1:], strict=True)
    ]
    max_regression = max(
        (max(value, 0.0) for value in adjacent_regressions), default=0.0
    )
    endpoint_passed = endpoint_improvement >= min_endpoint_improvement_percent
    adjacency_passed = max_regression <= max_adjacent_regression_percent
    return {
        "model_kind": model_kind,
        "split": split,
        "passed": endpoint_passed and adjacency_passed,
        "endpoint_passed": endpoint_passed,
        "adjacency_passed": adjacency_passed,
        "endpoint_improvement_percent": endpoint_improvement,
        "max_adjacent_regression_percent": max_regression,
        "required_endpoint_improvement_percent": min_endpoint_improvement_percent,
        "allowed_adjacent_regression_percent": max_adjacent_regression_percent,
        "curve": [
            {
                "constellation_size": run["constellation_size"],
                "coordinate_payload_bits": run["coordinate_payload_bits"],
                "chamfer_rmse": run[split]["chamfer_rmse"],
            }
            for run in curve
        ],
    }


def _run_summary(result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    parameter_ood = result["final_parameter_ood"]
    if parameter_ood is None:
        raise RuntimeError("Experiment 002 requires parameter-OOD evaluation")
    payload_bits = result["coordinate_payload_bits"]
    return {
        "model_kind": result["model_kind"],
        "constellation_size": config["constellation_size"],
        "bits_per_coordinate": config["bits"],
        "coordinate_payload_bits": payload_bits,
        "bits_per_input_point": payload_bits / config["num_points"],
        "parameter_count": result["parameter_count"],
        "encoder_parameter_count": result["encoder_parameter_count"],
        "decoder_parameter_count": result["decoder_parameter_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "validation": result["final_validation"],
        "parameter_ood": parameter_ood,
    }


def selected_rate_experiment(
    spec: SelectedRateSpec,
    *,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = Path(spec.base_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs = []
    combinations = [
        (size, model_kind)
        for size in spec.constellation_sizes
        for model_kind in spec.model_kinds
    ]

    for index, (size, model_kind) in enumerate(combinations, start=1):
        run_dir = output_dir / f"k{size}" / model_kind
        metrics_path = run_dir / "metrics.json"
        print(
            json.dumps(
                {
                    "experiment_point": index,
                    "total_points": len(combinations),
                    "constellation_size": size,
                    "model_kind": model_kind,
                }
            )
        )
        if resume and metrics_path.exists():
            result = json.loads(metrics_path.read_text())
        else:
            config = replace(
                spec.base_config,
                constellation_size=size,
                output_dir=str(run_dir),
            )
            result = train(config, device_name=device_name, model_kind=model_kind)
        runs.append(_run_summary(result))

    runs.sort(key=lambda run: (run["coordinate_payload_bits"], run["model_kind"]))
    gates = {
        split: rate_curve_gate(
            runs,
            model_kind=spec.gated_model_kind,
            split=split,
            min_endpoint_improvement_percent=(spec.min_endpoint_improvement_percent),
            max_adjacent_regression_percent=(spec.max_adjacent_regression_percent),
        )
        for split in ("validation", "parameter_ood")
    }
    result = {
        "base_config": asdict(spec.base_config),
        "constellation_sizes": list(spec.constellation_sizes),
        "model_kinds": list(spec.model_kinds),
        "gated_model_kind": spec.gated_model_kind,
        "runs": runs,
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates.values()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "selected_rate.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_002_relation_aware.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = selected_rate_experiment(
        SelectedRateSpec.from_json(args.config),
        device_name=args.device,
        resume=args.resume,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "runs"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
