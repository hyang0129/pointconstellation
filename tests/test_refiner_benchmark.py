"""Tests for the multi-seed Experiment 005 benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.refiner_benchmark import (
    RefinerBenchmarkConfig,
    paired_hierarchical_bootstrap,
    run_refiner_benchmark,
)


def test_paired_hierarchical_bootstrap_is_deterministic_and_paired() -> None:
    baseline = np.ones((3, 12), dtype=np.float64)
    method = np.full((3, 12), 0.81, dtype=np.float64)

    first = paired_hierarchical_bootstrap(
        baseline,
        method,
        samples=500,
        confidence_level=0.95,
        seed=19,
    )
    second = paired_hierarchical_bootstrap(
        baseline,
        method,
        samples=500,
        confidence_level=0.95,
        seed=19,
    )

    assert first == second
    assert first["relative_rmse_improvement_percent"] == pytest.approx(10.0)
    assert first["confidence_interval_lower_percent"] == pytest.approx(10.0)
    assert first["confidence_interval_upper_percent"] == pytest.approx(10.0)


def test_benchmark_config_rejects_nonreplicated_seed_list() -> None:
    with pytest.raises(ValueError, match="at least two unique"):
        RefinerBenchmarkConfig(model_seeds=(7,))


def test_tiny_multiseed_benchmark_runs_with_matched_controls(tmp_path) -> None:
    base_path = tmp_path / "base.json"
    base_path.write_text(
        json.dumps(
            {
                "num_points": 8,
                "input_sizes": [8],
                "constellation_sizes": [2],
                "bits": 8,
                "train_samples": 4,
                "validation_samples": 2,
                "parameter_ood_samples": 2,
                "batch_size": 2,
                "decoder_epochs": 1,
                "refiner_epochs": 1,
                "decoder_learning_rate": 0.001,
                "refiner_learning_rate": 0.001,
                "feature_width": 8,
                "num_heads": 2,
                "num_layers": 1,
                "recurrent_steps": 1,
                "responsibility_temperature": 0.2,
                "maximum_update": 0.1,
                "use_decoder_gradient": True,
                "seed": 7,
                "output_dir": str(tmp_path / "unused"),
            }
        )
    )
    config = RefinerBenchmarkConfig(
        base_experiment_config=str(base_path),
        model_seeds=(3, 5),
        data_seed=101,
        primary_input_size=8,
        primary_constellation_size=2,
        bootstrap_samples=100,
        output_dir=str(tmp_path / "benchmark"),
    )

    result = run_refiner_benchmark(config, device_name="cpu")
    saved = json.loads((tmp_path / "benchmark" / "benchmark_metrics.json").read_text())

    assert result["independently_trained_decoder_hashes"]
    assert len(saved["seed_runs"]) == 2
    assert all(run["matched_decoder"] for run in saved["seed_runs"])
    assert all(run["matched_refiner_initialization"] for run in saved["seed_runs"])
    assert all(run["matched_refiner_training_order"] for run in saved["seed_runs"])
    assert saved["dataset_manifests"]["validation"]["data_seed"] == 101
    for split in ("validation", "parameter_ood"):
        assert len(saved["splits"][split]["sample_ids"]) == 2
        assert "input_gradient_vs_no_feedback" in saved["splits"][split]
        assert len(saved["primary_convergence"][split]["input_gradient"]) == 2
        assert set(saved["splits"][split]["methods"]) == {
            "fps",
            "no_feedback_free_coordinates",
            "no_feedback_strict_projection",
            "input_gradient_free_coordinates",
            "input_gradient_strict_projection",
        }
        assert "median_per_cloud_rmse" in saved["splits"][split]["methods"]["fps"]
