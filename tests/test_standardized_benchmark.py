# ruff: noqa: E402

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.bitstream import expected_stream_bytes
from pointconstellation.codecs import Tmc3RatePoint
from pointconstellation.refiner_experiment import RefinerExperimentConfig
from pointconstellation.standardized_benchmark import (
    GpccBenchmarkConfig,
    StandardizedBenchmarkConfig,
    _interpolation_free_gpcc_comparisons,
    _pareto_frontier,
    run_standardized_benchmark,
)
from pointconstellation.standardized_metrics import (
    directed_nearest_squared,
    standardized_geometry_metrics,
)


def test_chunked_nearest_matches_full_pairwise() -> None:
    torch.manual_seed(1401)
    source = torch.rand(11, 3)
    target = torch.rand(13, 3)
    expected_distances, expected_indices = torch.cdist(source, target).square().min(1)

    distances, indices = directed_nearest_squared(source, target, chunk_size=3)

    assert torch.allclose(distances, expected_distances)
    assert torch.equal(indices, expected_indices)


def test_standardized_metrics_include_normal_and_tail_distortion() -> None:
    target = torch.tensor(
        [
            [-0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]]).expand_as(target)
    reconstruction = target + torch.tensor([0.0, 0.0, 0.1])

    metrics = standardized_geometry_metrics(
        reconstruction,
        target,
        normals,
        chunk_size=2,
        sliced_directions=8,
    )

    assert metrics["chamfer_rmse"] == pytest.approx(0.1)
    assert metrics["d2_mse_proxy"] == pytest.approx(0.01)
    assert metrics["hausdorff"] == pytest.approx(0.1)
    assert metrics["p99_euclidean"] == pytest.approx(0.1)
    assert metrics["d1_psnr_db_proxy"] > 20.0


def test_gpcc_common_and_rate_arguments_must_not_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        GpccBenchmarkConfig(
            executable="tmc3",
            encoder_args=("--positionQuantizationScale=1",),
            rate_points=(
                Tmc3RatePoint("duplicate", ("--positionQuantizationScale=0.5",)),
            ),
        )


def test_rate_comparison_marks_dominated_points_without_interpolation() -> None:
    gpcc = [
        {
            "split": "validation",
            "rate_point": "coarse",
            "actual_stream_bpp": 0.2,
            "chamfer_rmse": 0.3,
        },
        {
            "split": "validation",
            "rate_point": "dominated",
            "actual_stream_bpp": 0.25,
            "chamfer_rmse": 0.35,
        },
        {
            "split": "validation",
            "rate_point": "fine",
            "actual_stream_bpp": 0.3,
            "chamfer_rmse": 0.1,
        },
    ]
    neural = [
        {
            "split": "validation",
            "method": "free",
            "constellation_size": 8,
            "actual_stream_bpp": 0.25,
            "chamfer_rmse": 0.2,
        }
    ]

    frontier = _pareto_frontier(gpcc)
    comparison = _interpolation_free_gpcc_comparisons(neural, gpcc)[0]

    assert [row["rate_point"] for row in frontier] == ["coarse", "fine"]
    assert comparison["best_gpcc_at_or_below_rate"]["rate_point"] == "coarse"
    assert comparison["lowest_rate_gpcc_at_or_below_distortion"]["rate_point"] == (
        "fine"
    )
    assert not comparison["gpcc_measured_point_dominates"]


def test_tiny_standardized_benchmark_runs_through_bitstreams(tmp_path) -> None:
    experiment = RefinerExperimentConfig(
        num_points=16,
        input_sizes=(16,),
        constellation_sizes=(2, 4),
        bits=8,
        train_samples=4,
        validation_samples=2,
        parameter_ood_samples=2,
        batch_size=2,
        decoder_epochs=1,
        refiner_epochs=1,
        feature_width=8,
        num_heads=2,
        num_layers=1,
        recurrent_steps=1,
        run_internal_evaluation=False,
        seed=1403,
    )
    config = StandardizedBenchmarkConfig(
        experiment=experiment,
        profile_name="test",
        distance_chunk_size=8,
        sliced_directions=4,
        output_dir=str(tmp_path),
    )

    result = run_standardized_benchmark(config, device_name="cpu")
    saved = json.loads((tmp_path / "benchmark_metrics.json").read_text())
    lines = (tmp_path / "per_cloud.jsonl").read_text().splitlines()

    assert result["protocol"]["status"] == "protocol_aligned_procedural_proxy"
    assert not result["protocol"]["comparability_claims"]["shapenet"]
    assert result["model"]["decoder_unchanged"]
    assert len(lines) == 2 * 2 * 2 * 3
    assert len(saved["summary"]) == 2 * 2 * 3
    assert len(saved["monotonicity"]) == 2 * 3
    assert json.loads(lines[0])["stream_bytes"] in {
        expected_stream_bytes(2, 8),
        expected_stream_bytes(4, 8),
    }
    assert saved["config"]["experiment"]["run_internal_evaluation"] is False
    assert saved["manifests"]["validation"]["sha256"]

    resumed = run_standardized_benchmark(config, device_name="cpu", resume=True)
    assert resumed["model"]["reused_matching_checkpoint"]


def test_mesh_benchmark_uses_source_primary_and_independent_fresh_target(
    tmp_path,
) -> None:
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
        use_decoder_gradient=False,
        run_internal_evaluation=False,
        seed=1511,
    )
    config = StandardizedBenchmarkConfig(
        experiment=experiment,
        profile_name="mesh-test",
        evaluation_methods=("fps",),
        distance_chunk_size=8,
        sliced_directions=4,
        dataset_kind="mesh_manifest",
        dataset_root="tests/fixtures/meshes",
        dataset_manifest="tests/fixtures/mesh_manifest.json",
        output_dir=str(tmp_path),
    )

    result = run_standardized_benchmark(config, device_name="cpu")

    assert result["protocol"]["status"] == "external_mesh_surface_pilot"
    assert result["protocol"]["data_identity"]["training_target"] == "source"
    assert result["manifests"]["dataset"] == "pointconstellation-test-meshes"
    assert len(result["per_cloud"]) == 4
    assert all("fresh_chamfer_rmse" in row for row in result["per_cloud"])
    assert all(row["model_id"] for row in result["per_cloud"])
