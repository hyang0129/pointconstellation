#!/usr/bin/env python3
"""Generate the eleven missing Experiment 038 stability configurations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_ROOT / "configs" / "experiment_019_stability_modelnet40.json"
OBJECTIVE_CONFIG = PROJECT_ROOT / "configs" / "experiment_033_objective_sweep.json"
CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_REGIME = "k8_n2048"
REGIME_PATTERN = re.compile(r"k(?P<k>[0-9]+)_n(?P<n>[0-9]+)")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def _parse_regime(name: str) -> tuple[int, int]:
    match = REGIME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid regime name: {name}")
    return int(match.group("k")), int(match.group("n"))


def generate_configs(
    base: dict[str, Any], objective: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return path-keyed configs in the Experiment 033 declaration order."""

    regimes = objective.get("regimes")
    if not isinstance(regimes, list):
        raise ValueError("Experiment 033 config must contain a regimes list")
    generated: dict[str, dict[str, Any]] = {}
    seen_names: set[str] = set()
    for regime in regimes:
        if not isinstance(regime, dict):
            raise ValueError("Experiment 033 regimes must be JSON objects")
        name = str(regime["name"])
        if name in seen_names:
            raise ValueError(f"duplicate Experiment 033 regime: {name}")
        seen_names.add(name)
        constellation_size, num_points = _parse_regime(name)
        if constellation_size != int(regime["constellation_size"]):
            raise ValueError(f"regime name and K disagree: {name}")
        if num_points != int(regime["num_points"]):
            raise ValueError(f"regime name and N disagree: {name}")
        if name == BASE_REGIME:
            if (
                regime["stability_config"]
                != "configs/experiment_019_stability_modelnet40.json"
            ):
                raise ValueError(
                    "the base regime must continue to reference Experiment 019"
                )
            continue

        expected_config_path = f"configs/experiment_038_stability_{name}.json"
        expected_artifact_dir = f"artifacts/local/experiment_038_stability_{name}"
        if regime["stability_config"] != expected_config_path:
            raise ValueError(f"Experiment 033 config path differs for {name}")
        if regime["stability_artifact_dir"] != expected_artifact_dir:
            raise ValueError(f"Experiment 033 artifact path differs for {name}")

        config = dict(base)
        config["num_points"] = num_points
        config["constellation_size"] = constellation_size
        config["output_dir"] = expected_artifact_dir
        if num_points == int(base["num_points"]):
            config["decoder_source_artifact_dir"] = str(base["output_dir"])
        generated[expected_config_path] = config

    if len(generated) != 11:
        raise ValueError(f"expected eleven missing regimes, found {len(generated)}")
    return generated


def render_config(config: dict[str, Any]) -> str:
    """Render a stable, human-readable JSON representation."""

    return json.dumps(config, indent=4) + "\n"


def write_or_check_configs(
    generated: dict[str, dict[str, Any]], *, output_dir: Path, check: bool
) -> list[Path]:
    paths = []
    for relative, config in generated.items():
        path = output_dir / Path(relative).name
        expected = render_config(config)
        if check:
            if not path.is_file() or path.read_text() != expected:
                raise RuntimeError(f"generated config is stale or absent: {path}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--objective-config", type=Path, default=OBJECTIVE_CONFIG)
    parser.add_argument("--output-config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if any checked-in generated config differs",
    )
    args = parser.parse_args()
    generated = generate_configs(
        _read_json(args.base_config), _read_json(args.objective_config)
    )
    paths = write_or_check_configs(
        generated, output_dir=args.output_config_dir, check=args.check
    )
    print(
        json.dumps(
            {
                "status": "checked" if args.check else "written",
                "configs": [str(path) for path in paths],
            }
        )
    )


if __name__ == "__main__":
    main()
