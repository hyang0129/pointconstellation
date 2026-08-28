from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pointconstellation.benchmark_registry import (
    build_registry,
    collect_registry_rows,
    headline_statistics,
    load_registry,
)

FIXTURES = Path("tests/fixtures/benchmark_artifacts")


def test_registry_fixture_has_expected_rows_and_stable_order(tmp_path: Path) -> None:
    output = tmp_path / "benchmark_registry.jsonl"

    rows = build_registry(FIXTURES, output)
    first_bytes = output.read_bytes()
    rebuilt = build_registry(FIXTURES, output)

    assert len(rows) == 28
    assert rows == rebuilt == load_registry(output)
    assert output.read_bytes() == first_bytes
    assert [(row["split"], row["method"], row["metric_name"]) for row in rows[:4]] == [
        ("ood", "fps", "official_d1_mse"),
        ("ood", "fps", "official_d2_mse"),
        ("ood", "fps", "official_d1_mse"),
        ("ood", "fps", "official_d2_mse"),
    ]
    assert rows[-1]["method"] == "refiner"
    assert rows[-1]["metric_name"] == "official_d2_mse"
    adam = next(row for row in rows if row["method"] == "adam_16")
    assert adam["arm_label"] == "stabilized"
    assert adam["rate_bytes"] == 50.0
    source = Path(adam["source_path"])
    assert adam["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    official = next(
        row
        for row in rows
        if row["experiment"].startswith("experiment_020")
        and row["metric_name"] == "official_d1_mse"
    )
    assert official["arm_label"] == "stabilized"


def test_headline_statistics_use_latest_paired_official_rows(tmp_path: Path) -> None:
    rows = build_registry(FIXTURES, tmp_path / "registry.jsonl")

    statistics = headline_statistics(rows, bootstrap_samples=100)

    validation = statistics["panels"]["validation"]["methods"]
    ood = statistics["panels"]["ood"]["methods"]
    assert statistics["dataset"] == "FixtureNet"
    assert set(validation) == {"fps", "refiner"}
    assert validation["fps"]["rmse"] == pytest.approx(math.sqrt(122.0))
    assert validation["refiner"]["rmse"] == pytest.approx(math.sqrt(72.5))
    assert ood["fps"]["rmse"] == pytest.approx(math.sqrt(226.0))
    assert ood["refiner"]["rmse"] == pytest.approx(math.sqrt(145.0))
    assert all(value["paired_units"] == 2 for value in validation.values())


def test_future_experiment_jsonl_is_discovered_when_present(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment_022_selection_baselines"
    experiment.mkdir()
    (experiment / "selection_metrics.json").write_text(
        json.dumps(
            {
                "dataset": "FixtureNet",
                "config": {"num_points": 2048, "decoder_arm": "stabilized"},
            }
        )
    )
    source = experiment / "selection_per_cloud.jsonl"
    source.write_text(
        json.dumps(
            {
                "split": "validation",
                "sample_id": 0,
                "family": "chair",
                "model_id": "chair_0001",
                "method": "random-best-of-64",
                "stream_bytes": 50,
                "d1_mse": 90.0,
            }
        )
        + "\n"
    )

    rows = collect_registry_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["method"] == "random_best_of_n"
    assert rows[0]["arm_label"] == "stabilized"
    assert rows[0]["rate_bpp"] == pytest.approx(0.1953125)
    assert rows[0]["metric_name"] == "official_d1_mse"
    assert rows[0]["source_path"] == source.as_posix()


def test_experiment_025_preserves_payload_and_normalization_rates(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment_025_rate_curve"
    experiment.mkdir()
    (experiment / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "FixtureNet",
                "config": {"num_points": 1000},
            }
        )
    )
    source = experiment / "rate_per_cloud.jsonl"
    source.write_text(
        json.dumps(
            {
                "split": "validation",
                "sample_id": 0,
                "family": "chair",
                "model_id": "chair_0001",
                "method": "refiner",
                "stream_bytes": 64,
                "header_bytes": 14,
                "payload_bytes": 36,
                "normalization_metadata_bytes": 14,
                "amortized_stream_bytes": 50.5,
                "d1_psnr_db": 32.0,
            }
        )
        + "\n"
    )

    rows = collect_registry_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["experiment"] == "experiment_025_rate_curve"
    assert rows[0]["metric_name"] == "official_d1_psnr_db"
    assert rows[0]["payload_bytes"] == 36.0
    assert rows[0]["payload_bpp"] == pytest.approx(0.288)
    assert rows[0]["normalization_bytes"] == 14.0
    assert rows[0]["normalization_bpp"] == pytest.approx(0.112)
    assert rows[0]["amortized_stream_bpp"] == pytest.approx(0.404)


def test_figure_script_runs_headless_on_fixture_registry(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    registry = tmp_path / "registry.jsonl"
    build_registry(FIXTURES, registry)
    output_dir = tmp_path / "figures"
    environment = {**os.environ, "MPLBACKEND": "Agg"}

    result = subprocess.run(
        [
            sys.executable,
            "scripts/figures/fig1_selection_baselines.py",
            "--registry",
            str(registry),
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "fig1_selection_baselines.pdf").stat().st_size > 0
    assert (output_dir / "fig1_selection_baselines.png").stat().st_size > 0


def test_table_script_uses_only_registry(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    build_registry(FIXTURES, registry)
    output = tmp_path / "table.tex"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/figures/table1_headline.py",
            "--registry",
            str(registry),
            "--output",
            str(output),
            "--bootstrap-samples",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Official D1 RMSE" in output.read_text()
    assert json.loads(result.stdout)["table"] == str(output)


def test_rd_figure_runs_headless_on_fixture_registry(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    registry = tmp_path / "registry.jsonl"
    build_registry(FIXTURES, registry)
    output_dir = tmp_path / "figures"
    environment = {**os.environ, "MPLBACKEND": "Agg"}

    figure_result = subprocess.run(
        [
            sys.executable,
            "scripts/figures/fig_rd_positioning.py",
            "--registry",
            str(registry),
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert figure_result.returncode == 0, figure_result.stderr
    assert (output_dir / "fig_rd_positioning.pdf").stat().st_size > 0
    assert (output_dir / "fig_rd_positioning.png").stat().st_size > 0


def test_bd_table_runs_on_fixture_registry(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    build_registry(FIXTURES, registry)
    table = tmp_path / "table_bd_rate.tex"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/figures/table_bd_rate.py",
            "--registry",
            str(registry),
            "--output",
            str(table),
            "--bootstrap-samples",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert r"\textit{insufficient overlap}" in table.read_text()
