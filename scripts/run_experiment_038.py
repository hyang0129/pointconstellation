#!/usr/bin/env python3
"""Run one Experiment 038 N/K cell through stability and official metrics."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.official_stability import (  # noqa: E402
    OfficialStabilityConfig,
    run_official_stability,
)
from pointconstellation.stability_experiment import (  # noqa: E402
    StabilityExperimentConfig,
    run_stability_experiment,
)

REGIME_PATTERN = re.compile(r"k(?P<k>[0-9]+)_n(?P<n>[0-9]+)")
REGIMES = (
    "k4_n1024",
    "k8_n1024",
    "k16_n1024",
    "k32_n1024",
    "k4_n2048",
    "k16_n2048",
    "k32_n2048",
    "k4_n4096",
    "k8_n4096",
    "k16_n4096",
    "k32_n4096",
)


def _parse_regime(name: str) -> tuple[int, int]:
    match = REGIME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid regime name: {name}")
    return int(match.group("k")), int(match.group("n"))


def _smoke_config(
    config: StabilityExperimentConfig, *, output_dir: Path
) -> StabilityExperimentConfig:
    """Derive the labeled eight-cloud, one-seed CPU smoke protocol."""

    return replace(
        config,
        dataset_kind="procedural",
        dataset_root=None,
        dataset_manifest=None,
        calibration_split="train",
        verify_mesh_hashes=False,
        train_samples=2,
        calibration_samples=2,
        validation_samples=2,
        ood_samples=2,
        batch_size=2,
        decoder_seeds=(config.decoder_seeds[0],),
        refiner_seeds=(config.refiner_seeds[0],),
        stabilized_decoder_epochs=2,
        feature_width=8,
        num_heads=2,
        recurrent_steps=1,
        distance_chunk_size=128,
        adam_probe_evaluations=0,
        adam_probe_clouds_per_split=0,
        reference_feature_dir=None,
        expected_manifest_sha256=None,
        decoder_source_artifact_dir=None,
        allow_single_seed=True,
        bootstrap_samples=100,
        output_dir=str(output_dir),
    )


def _acquire_lock(output_dir: Path) -> TextIO:
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = (output_dir / ".experiment_038.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"regime output is already locked: {output_dir}") from error
    return handle


def _completed_stability(
    path: Path, config: StabilityExperimentConfig
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text())
    expected = json.loads(json.dumps(asdict(config)))
    if metrics.get("config") != expected:
        raise RuntimeError("existing Experiment 038 stability config differs")
    checks = metrics.get("contract_checks")
    if not metrics.get("factorial", {}).get("complete") or not isinstance(checks, dict):
        raise RuntimeError("existing Experiment 038 stability artifact is incomplete")
    if not checks or not all(checks.values()):
        raise RuntimeError("existing Experiment 038 stability contract failed")
    return metrics


def run_regime(
    regime: str,
    *,
    device: str,
    pc_error: Path,
    output_dir: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run or safely resume one declared Experiment 038 regime."""

    if regime not in REGIMES:
        raise ValueError(f"unknown Experiment 038 regime: {regime}")
    constellation_size, num_points = _parse_regime(regime)
    config_path = PROJECT_ROOT / "configs" / f"experiment_038_stability_{regime}.json"
    config = StabilityExperimentConfig.from_json(config_path)
    if (config.constellation_size, config.num_points) != (
        constellation_size,
        num_points,
    ):
        raise RuntimeError(f"generated config does not match regime {regime}")
    destination = output_dir or Path(config.output_dir)
    if output_dir is not None:
        config = replace(config, output_dir=str(destination))
    if smoke:
        config = _smoke_config(config, output_dir=destination)
    lock = _acquire_lock(destination)
    started = time.perf_counter()
    try:
        effective_config_path = config_path
        if smoke or output_dir is not None:
            effective_config_path = destination / "effective_stability_config.json"
            effective_config_path.write_text(
                json.dumps(asdict(config), indent=2) + "\n"
            )

        stability_path = destination / "stability_metrics.json"
        stability = _completed_stability(stability_path, config)
        stability_resumed = stability is not None
        if stability is None:
            partial_paths = (destination / "decoders", destination / "pairs")
            if any(path.exists() for path in partial_paths):
                raise RuntimeError(
                    "partial stability output exists; choose a clean output directory"
                )
            stability = run_stability_experiment(
                config,
                device_name=device,
                experiment_name="038_stabilized_decoder_regime",
            )

        official_output = destination / "official"
        official = OfficialStabilityConfig(
            stability_config=str(effective_config_path),
            stability_artifact_dir=str(destination),
            pc_error_executable=str(pc_error),
            position_bits=config.coordinate_bits,
            decoder_seeds=config.decoder_seeds,
            refiner_seeds=config.refiner_seeds,
            splits=("validation", "ood"),
            max_clouds_per_split=2 if smoke else None,
            allow_single_seed=smoke,
            bootstrap_samples=100 if smoke else config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed + 1,
            confidence_level=config.confidence_level,
            output_dir=str(official_output),
        )
        official_result = run_official_stability(official, device_name=device)
        result = {
            "experiment": "038_stabilized_decoder_regime",
            "regime": regime,
            "num_points": num_points,
            "constellation_size": constellation_size,
            "coordinate_bits": config.coordinate_bits,
            "device": device,
            "smoke": smoke,
            "single_seed_smoke_is_inferential": False if smoke else None,
            "decoder_source_artifact_dir": config.decoder_source_artifact_dir,
            "decoder_training_reused": bool(config.decoder_source_artifact_dir),
            "stability_resumed": stability_resumed,
            "stability_elapsed_seconds": stability["elapsed_seconds"],
            "official_elapsed_seconds": official_result["elapsed_seconds"],
            "wall_clock_seconds": time.perf_counter() - started,
            "stability_metrics": str(stability_path),
            "official_metrics": str(official_output / "official_metrics.json"),
        }
        (destination / "experiment_038_run.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result))
        return result
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=REGIMES)
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument(
        "--pc-error",
        type=Path,
        default=Path("artifacts/tools/mpeg-pcc-dmetric/build/Release/pc_error"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a labeled eight-cloud, one-decoder/refiner-seed protocol",
    )
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    run_regime(
        args.regime,
        device=args.device,
        pc_error=args.pc_error,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
