from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pointconstellation.benchmark_registry import (
    build_registry,
    rate_utility_statistics,
)

FIXTURES = Path("tests/fixtures/rate_utility_artifacts")


def test_track_b_registry_ingestion_preserves_metrics_and_dimensions(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.jsonl"

    rows = build_registry(FIXTURES, registry)

    assert len(rows) == 30
    assert {row["metric_name"] for row in rows} == {
        "accuracy",
        "map",
        "normal_consistency",
        "official_d1_mse",
        "official_d1_rmse",
        "repeatability",
        "surface_rmse",
    }
    coordinate = next(
        row
        for row in rows
        if row["arm_label"] == "coordinate_only"
        and row["rate_bytes"] == 50.0
        and row["metric_name"] == "accuracy"
    )
    assert coordinate["representation_family"] == "coordinate_set"
    assert coordinate["objective"] == "geometry_only"
    assert coordinate["regime"] == "frozen_probe"
    assert coordinate["record_kind"] == "aggregate"
    assert coordinate["rate_bpp"] == pytest.approx(0.1953125)
    source = Path(coordinate["source_path"])
    assert (
        coordinate["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    )

    joint = next(row for row in rows if row["arm_label"] == "joint_utility")
    assert joint["experiment"].startswith("experiment_034")
    assert joint["representation_family"] == "coordinate_set"
    assert joint["objective"] == "joint_utility"
    assert joint["regime"] == "end_to_end"
    assert joint["run_seed"] == 17


def test_rate_utility_statistics_aggregate_50_byte_arms(tmp_path: Path) -> None:
    rows = build_registry(FIXTURES, tmp_path / "registry.jsonl")

    statistics = rate_utility_statistics(rows, rate_bytes=50.0)

    assert statistics["dataset"] == "FixtureNet"
    assert statistics["split"] == "validation"
    assert len(statistics["points"]) == 3
    points = {point["arm_label"]: point for point in statistics["points"]}
    assert points["coordinate_only"]["d1_rmse"] == pytest.approx(10.0)
    assert points["feature_baseline"]["d1_rmse"] == pytest.approx(11.0)
    assert points["feature_baseline"]["accuracy"] == pytest.approx(0.84)
    assert points["joint_utility"]["d1_rmse"] == pytest.approx(90.0**0.5)


@pytest.mark.parametrize(
    ("script", "stem"),
    (
        ("scripts/figures/fig_rate_utility.py", "fig_rate_utility"),
        ("scripts/figures/fig_objective_pareto.py", "fig_objective_pareto"),
    ),
)
def test_rate_utility_figures_run_headless(
    tmp_path: Path, script: str, stem: str
) -> None:
    pytest.importorskip("matplotlib")
    registry = tmp_path / "registry.jsonl"
    build_registry(FIXTURES, registry)
    output_dir = tmp_path / "figures"
    environment = {**os.environ, "MPLBACKEND": "Agg"}

    result = subprocess.run(
        [
            sys.executable,
            script,
            "--registry",
            str(registry),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / f"{stem}.pdf").stat().st_size > 0
    assert (output_dir / f"{stem}.png").stat().st_size > 0
    assert set(json.loads(result.stdout)) == {"pdf", "png"}


def test_representation_table_includes_all_50_byte_arms(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    build_registry(FIXTURES, registry)
    output = tmp_path / "table_representation.tex"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/figures/table_representation.py",
            "--registry",
            str(registry),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    table = output.read_text()
    assert "coordinate\\_only" in table
    assert "feature\\_baseline" in table
    assert "joint\\_utility" in table
    assert "Representation metrics at 50 serialized bytes" in table
    assert json.loads(result.stdout) == {"table": str(output)}
