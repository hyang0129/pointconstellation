"""Run a matched-rate learned-constellation versus FPS comparison."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pointconstellation.train import TrainingConfig, train


def compare(
    config: TrainingConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs: dict[str, Any] = {}

    for model_kind in ("learned", "fps"):
        run_config = replace(config, output_dir=str(output_dir / model_kind))
        runs[model_kind] = train(
            run_config,
            device_name=device_name,
            model_kind=model_kind,
        )

    learned_rmse = runs["learned"]["final_validation"]["chamfer_rmse"]
    fps_rmse = runs["fps"]["final_validation"]["chamfer_rmse"]
    result = {
        "config": asdict(config),
        "coordinate_payload_bits": 3 * config.constellation_size * config.bits,
        "learned": {
            "parameter_count": runs["learned"]["parameter_count"],
            "encoder_parameter_count": runs["learned"]["encoder_parameter_count"],
            "decoder_parameter_count": runs["learned"]["decoder_parameter_count"],
            "initial_validation": runs["learned"]["initial_validation"],
            "final_validation": runs["learned"]["final_validation"],
            "elapsed_seconds": runs["learned"]["elapsed_seconds"],
        },
        "fps": {
            "parameter_count": runs["fps"]["parameter_count"],
            "encoder_parameter_count": runs["fps"]["encoder_parameter_count"],
            "decoder_parameter_count": runs["fps"]["decoder_parameter_count"],
            "initial_validation": runs["fps"]["initial_validation"],
            "final_validation": runs["fps"]["final_validation"],
            "elapsed_seconds": runs["fps"]["elapsed_seconds"],
        },
        "winner": "learned" if learned_rmse < fps_rmse else "fps",
        "learned_rmse_reduction_vs_fps_percent": 100.0
        * (fps_rmse - learned_rmse)
        / fps_rmse,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_001_fps_comparison.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    args = parser.parse_args()

    result = compare(TrainingConfig.from_json(args.config), device_name=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
