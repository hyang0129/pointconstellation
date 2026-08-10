"""Run Experiment 003's matched-decoder encoder isolation."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from pointconstellation.train import TrainingConfig, train


@dataclass(frozen=True)
class EncoderIsolationSpec:
    base_config: TrainingConfig
    projection_temperatures: tuple[float, ...]
    surface_weights: tuple[float, ...]
    max_surface_rmse: float = 0.01
    max_rmse_gap_vs_fps_percent: float = 5.0

    def __post_init__(self) -> None:
        if not self.projection_temperatures or not self.surface_weights:
            raise ValueError("projection and surface sweep axes must not be empty")
        if len(set(self.projection_temperatures)) != len(self.projection_temperatures):
            raise ValueError("projection_temperatures must be unique")
        if len(set(self.surface_weights)) != len(self.surface_weights):
            raise ValueError("surface_weights must be unique")
        if any(value <= 0 for value in self.projection_temperatures):
            raise ValueError("projection temperatures must be positive")
        if any(value < 0 for value in self.surface_weights):
            raise ValueError("surface weights cannot be negative")
        if self.base_config.parameter_ood_samples < 1:
            raise ValueError("parameter_ood_samples must be positive")
        if self.max_surface_rmse <= 0:
            raise ValueError("max_surface_rmse must be positive")
        if self.max_rmse_gap_vs_fps_percent < 0:
            raise ValueError("max_rmse_gap_vs_fps_percent cannot be negative")

    @classmethod
    def from_json(cls, path: Path) -> EncoderIsolationSpec:
        values = json.loads(path.read_text())
        return cls(
            base_config=TrainingConfig(**values["base_config"]),
            projection_temperatures=tuple(values["projection_temperatures"]),
            surface_weights=tuple(values["surface_weights"]),
            max_surface_rmse=values.get("max_surface_rmse", 0.01),
            max_rmse_gap_vs_fps_percent=values.get("max_rmse_gap_vs_fps_percent", 5.0),
        )


def _float_slug(value: float) -> str:
    return format(value, ".6g").replace(".", "p").replace("-", "m")


def _conditions(spec: EncoderIsolationSpec) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = [
        {
            "condition": "fps",
            "model_kind": "relation_fps",
            "projection_temperature": spec.base_config.projection_temperature,
            "surface_weight": spec.base_config.surface_weight,
        },
        {
            "condition": "hard_subset",
            "model_kind": "relation_subset",
            "projection_temperature": spec.base_config.projection_temperature,
            "surface_weight": spec.base_config.surface_weight,
        },
    ]
    conditions.extend(
        {
            "condition": (
                f"soft_t{_float_slug(temperature)}_s{_float_slug(surface_weight)}"
            ),
            "model_kind": "relation",
            "projection_temperature": temperature,
            "surface_weight": surface_weight,
        }
        for temperature in spec.projection_temperatures
        for surface_weight in spec.surface_weights
    )
    return conditions


def _run_summary(condition: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    parameter_ood = result["final_parameter_ood"]
    if parameter_ood is None:
        raise RuntimeError("Experiment 003 requires parameter-OOD evaluation")
    return {
        **condition,
        "constellation_size": result["config"]["constellation_size"],
        "bits_per_coordinate": result["config"]["bits"],
        "coordinate_payload_bits": result["coordinate_payload_bits"],
        "parameter_count": result["parameter_count"],
        "encoder_parameter_count": result["encoder_parameter_count"],
        "decoder_parameter_count": result["decoder_parameter_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "validation": result["final_validation"],
        "parameter_ood": parameter_ood,
    }


def encoder_isolation_gate(
    runs: list[dict[str, Any]],
    *,
    max_surface_rmse: float,
    max_rmse_gap_vs_fps_percent: float,
) -> dict[str, Any]:
    """Select on validation only, then evaluate the selected candidate on OOD."""

    references = [run for run in runs if run["condition"] == "fps"]
    if len(references) != 1:
        raise ValueError("exactly one FPS reference is required")
    reference = references[0]
    candidates = [run for run in runs if run["condition"] != "fps"]
    eligible = [
        run
        for run in candidates
        if math.sqrt(run["validation"]["surface"]) <= max_surface_rmse
    ]
    if not eligible:
        return {
            "passed": False,
            "reason": "no learned candidate met the validation surface constraint",
            "selected_condition": None,
            "max_surface_rmse": max_surface_rmse,
            "max_rmse_gap_vs_fps_percent": max_rmse_gap_vs_fps_percent,
        }

    selected = min(eligible, key=lambda run: run["validation"]["chamfer_rmse"])
    validation_gap = (
        100.0
        * (
            selected["validation"]["chamfer_rmse"]
            - reference["validation"]["chamfer_rmse"]
        )
        / reference["validation"]["chamfer_rmse"]
    )
    ood_gap = (
        100.0
        * (
            selected["parameter_ood"]["chamfer_rmse"]
            - reference["parameter_ood"]["chamfer_rmse"]
        )
        / reference["parameter_ood"]["chamfer_rmse"]
    )
    validation_surface_rmse = math.sqrt(selected["validation"]["surface"])
    ood_surface_rmse = math.sqrt(selected["parameter_ood"]["surface"])
    checks = {
        "validation_gap": validation_gap <= max_rmse_gap_vs_fps_percent,
        "parameter_ood_gap": ood_gap <= max_rmse_gap_vs_fps_percent,
        "validation_surface": validation_surface_rmse <= max_surface_rmse,
        "parameter_ood_surface": ood_surface_rmse <= max_surface_rmse,
    }
    return {
        "passed": all(checks.values()),
        "reason": None,
        "selected_condition": selected["condition"],
        "checks": checks,
        "max_surface_rmse": max_surface_rmse,
        "max_rmse_gap_vs_fps_percent": max_rmse_gap_vs_fps_percent,
        "validation_rmse_gap_vs_fps_percent": validation_gap,
        "parameter_ood_rmse_gap_vs_fps_percent": ood_gap,
        "validation_surface_rmse": validation_surface_rmse,
        "parameter_ood_surface_rmse": ood_surface_rmse,
        "selected_validation_rmse": selected["validation"]["chamfer_rmse"],
        "selected_parameter_ood_rmse": selected["parameter_ood"]["chamfer_rmse"],
        "fps_validation_rmse": reference["validation"]["chamfer_rmse"],
        "fps_parameter_ood_rmse": reference["parameter_ood"]["chamfer_rmse"],
    }


def encoder_isolation_experiment(
    spec: EncoderIsolationSpec,
    *,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = Path(spec.base_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = _conditions(spec)
    runs = []
    started = time.perf_counter()

    for index, condition in enumerate(conditions, start=1):
        run_dir = output_dir / condition["condition"]
        metrics_path = run_dir / "metrics.json"
        print(
            json.dumps(
                {
                    "experiment_point": index,
                    "total_points": len(conditions),
                    **condition,
                }
            )
        )
        if resume and metrics_path.exists():
            trained = json.loads(metrics_path.read_text())
        else:
            config = replace(
                spec.base_config,
                projection_temperature=condition["projection_temperature"],
                surface_weight=condition["surface_weight"],
                output_dir=str(run_dir),
            )
            trained = train(
                config,
                device_name=device_name,
                model_kind=condition["model_kind"],
            )
        runs.append(_run_summary(condition, trained))

    gate = encoder_isolation_gate(
        runs,
        max_surface_rmse=spec.max_surface_rmse,
        max_rmse_gap_vs_fps_percent=spec.max_rmse_gap_vs_fps_percent,
    )
    result = {
        "base_config": asdict(spec.base_config),
        "projection_temperatures": list(spec.projection_temperatures),
        "surface_weights": list(spec.surface_weights),
        "runs": runs,
        "validation_ranking": [
            {
                "condition": run["condition"],
                "chamfer_rmse": run["validation"]["chamfer_rmse"],
                "anchor_surface_rmse": math.sqrt(run["validation"]["surface"]),
            }
            for run in sorted(runs, key=lambda run: run["validation"]["chamfer_rmse"])
        ],
        "gate": gate,
        "passed": gate["passed"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "encoder_isolation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_003_encoder_isolation.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = encoder_isolation_experiment(
        EncoderIsolationSpec.from_json(args.config),
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
