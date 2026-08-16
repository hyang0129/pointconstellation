# ruff: noqa: E402

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

pytest.importorskip("torch")

from pointconstellation.refiner_experiment import RefinerExperimentConfig
from pointconstellation.standardized_benchmark import StandardizedBenchmarkConfig
from pointconstellation.standardized_multiseed import (
    StandardizedMultiSeedConfig,
    run_standardized_multiseed,
)


def test_tiny_mesh_multiseed_run_is_paired_and_independent(tmp_path) -> None:
    experiment = RefinerExperimentConfig(
        num_points=16,
        input_sizes=(16,),
        constellation_sizes=(2,),
        bits=8,
        train_samples=2,
        validation_samples=2,
        parameter_ood_samples=2,
        batch_size=1,
        decoder_epochs=1,
        refiner_epochs=1,
        feature_width=8,
        num_heads=2,
        num_layers=1,
        recurrent_steps=1,
        use_decoder_gradient=True,
        decoder_gradient_chunk_size=8,
        run_internal_evaluation=False,
        seed=3,
    )
    benchmark = StandardizedBenchmarkConfig(
        experiment=experiment,
        profile_name="multiseed-test",
        distance_chunk_size=8,
        sliced_directions=4,
        dataset_kind="mesh_manifest",
        dataset_root="tests/fixtures/meshes",
        dataset_manifest="tests/fixtures/mesh_manifest.json",
        output_dir=str(tmp_path / "unused"),
    )
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(asdict(benchmark)))
    config = StandardizedMultiSeedConfig(
        base_benchmark_config=str(base_path),
        model_seeds=(3, 5),
        data_seed=43,
        primary_constellation_size=2,
        bootstrap_samples=100,
        output_dir=str(tmp_path / "run"),
    )

    result = run_standardized_multiseed(config, device_name="cpu")
    saved = json.loads((tmp_path / "run/multiseed_metrics.json").read_text())

    assert result["data_identity"]["training_target"] == "source"
    assert result["model_independence"]["all_unique"]
    assert len(set(result["model_independence"]["decoder_hashes"])) == 2
    assert len(set(result["model_independence"]["refiner_hashes"])) == 2
    assert len(result["paired_comparisons"]) == 2 * 1 * 2
    assert len(result["per_family_comparisons"]) == 2 * 1 * 2
    assert len(result["representative_examples"]) == 2 * 2 * 2 * 2
    assert all(
        comparison["metrics"]["chamfer_mse"]["seed_count"] == 2
        for comparison in result["paired_comparisons"]
    )
    assert saved["primary_gate"]["constellation_size"] == 2
    assert saved["gpcc"] is None

    aggregated = run_standardized_multiseed(
        config, device_name="cpu", aggregate_only=True
    )
    assert aggregated["model_independence"] == result["model_independence"]
    assert aggregated["elapsed_seconds"] == result["elapsed_seconds"]
