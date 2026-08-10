"""Sweep constellation cardinality and coordinate precision at matched rates."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from pointconstellation.compare import compare
from pointconstellation.train import TrainingConfig


@dataclass(frozen=True)
class SweepSpec:
    base_config: TrainingConfig
    constellation_sizes: tuple[int, ...]
    coordinate_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.constellation_sizes or not self.coordinate_bits:
            raise ValueError("sweep axes must not be empty")
        if len(set(self.constellation_sizes)) != len(self.constellation_sizes):
            raise ValueError("constellation_sizes must be unique")
        if len(set(self.coordinate_bits)) != len(self.coordinate_bits):
            raise ValueError("coordinate_bits must be unique")
        if any(
            size < 2 or size > self.base_config.num_points
            for size in self.constellation_sizes
        ):
            raise ValueError("constellation sizes must be between 2 and num_points")
        if any(bits < 2 or bits > 24 for bits in self.coordinate_bits):
            raise ValueError("coordinate bits must be between 2 and 24")

    @classmethod
    def from_json(cls, path: Path) -> SweepSpec:
        values = json.loads(path.read_text())
        return cls(
            base_config=TrainingConfig(**values["base_config"]),
            constellation_sizes=tuple(values["constellation_sizes"]),
            coordinate_bits=tuple(values["coordinate_bits"]),
        )


def pareto_frontier(
    points: list[dict[str, Any]], model_kind: str
) -> list[dict[str, Any]]:
    """Return rate-sorted points that strictly improve distortion."""

    frontier = []
    best_rmse = float("inf")
    for point in sorted(points, key=lambda item: item["coordinate_payload_bits"]):
        rmse = point[model_kind]["chamfer_rmse"]
        if rmse < best_rmse:
            frontier.append(
                {
                    "constellation_size": point["constellation_size"],
                    "bits_per_coordinate": point["bits_per_coordinate"],
                    "coordinate_payload_bits": point["coordinate_payload_bits"],
                    "bits_per_input_point": point["bits_per_input_point"],
                    "chamfer_rmse": rmse,
                }
            )
            best_rmse = rmse
    return frontier


def _point_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    config = comparison["config"]
    learned = comparison["learned"]["final_validation"]
    fps = comparison["fps"]["final_validation"]
    payload_bits = comparison["coordinate_payload_bits"]
    return {
        "constellation_size": config["constellation_size"],
        "bits_per_coordinate": config["bits"],
        "coordinate_payload_bits": payload_bits,
        "bits_per_input_point": payload_bits / config["num_points"],
        "input_to_constellation_point_ratio": config["num_points"]
        / config["constellation_size"],
        "learned": {
            "chamfer_rmse": learned["chamfer_rmse"],
            "anchor_surface_rmse": learned["surface"] ** 0.5,
        },
        "fps": {
            "chamfer_rmse": fps["chamfer_rmse"],
            "anchor_surface_rmse": fps["surface"] ** 0.5,
        },
        "winner": comparison["winner"],
        "learned_rmse_reduction_vs_fps_percent": comparison[
            "learned_rmse_reduction_vs_fps_percent"
        ],
    }


def sweep(
    spec: SweepSpec,
    *,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = Path(spec.base_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    points = []
    combinations = [
        (size, bits)
        for size in spec.constellation_sizes
        for bits in spec.coordinate_bits
    ]

    for index, (size, bits) in enumerate(combinations, start=1):
        point_dir = output_dir / f"k{size}_q{bits}"
        comparison_path = point_dir / "comparison.json"
        print(
            json.dumps(
                {
                    "sweep_point": index,
                    "total_points": len(combinations),
                    "constellation_size": size,
                    "bits_per_coordinate": bits,
                }
            )
        )
        if resume and comparison_path.exists():
            comparison = json.loads(comparison_path.read_text())
        else:
            config = replace(
                spec.base_config,
                constellation_size=size,
                bits=bits,
                output_dir=str(point_dir),
            )
            comparison = compare(config, device_name=device_name)
        points.append(_point_summary(comparison))

    points.sort(key=lambda item: item["coordinate_payload_bits"])
    result = {
        "base_config": asdict(spec.base_config),
        "constellation_sizes": list(spec.constellation_sizes),
        "coordinate_bits": list(spec.coordinate_bits),
        "runs": points,
        "winner_counts": {
            model_kind: sum(point["winner"] == model_kind for point in points)
            for model_kind in ("learned", "fps")
        },
        "pareto_frontier": {
            model_kind: pareto_frontier(points, model_kind)
            for model_kind in ("learned", "fps")
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_001_rate_sweep.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = sweep(
        SweepSpec.from_json(args.config),
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
