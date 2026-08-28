"""Focused tests for Experiment 031 procedural Gate B."""

# ruff: noqa: E402

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.data import (
    ProceduralSurface,
    ProceduralSurfaceDataset,
    analytic_surface_distances,
    generate_surface_triplet,
)
from pointconstellation.geometry_gate_experiment import (
    GeometryGateExperimentConfig,
    perturb_quantized_coordinates,
    run_geometry_gate_experiment,
)
from pointconstellation.quantization import quantize_coordinates


def _surface(family: str, parameters: tuple[float, ...]) -> ProceduralSurface:
    return ProceduralSurface(
        family=family,
        parameters=parameters,
        rotation=np.eye(3, dtype=np.float32),
        scale=1.0,
        sample_id=0,
        split="validation",
    )


def test_analytic_distance_on_known_plane_and_sphere() -> None:
    plane = _surface("plane", (1.0, 1.0))
    sphere = _surface("sphere", (1.0,))

    plane_distances = analytic_surface_distances(
        [[0.2, -0.4, 0.0], [0.0, 0.0, 0.5], [1.3, 1.4, 0.0]], plane
    )
    sphere_distances = analytic_surface_distances(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.5], [0.0, 0.0, 0.0]], sphere
    )

    assert plane_distances == pytest.approx([0.0, 0.5, 0.5])
    assert sphere_distances == pytest.approx([0.0, 0.5, 1.0])


def test_surface_triplet_is_deterministic_independent_and_on_surface() -> None:
    first = generate_surface_triplet(3, num_points=64, seed=37, split="validation")
    repeated = generate_surface_triplet(3, num_points=64, seed=37, split="validation")

    assert np.array_equal(first.x_a.points, repeated.x_a.points)
    assert not np.array_equal(first.x_a.points, first.x_b.points)
    assert not np.array_equal(first.x_b.points, first.x_c.points)
    for role in (first.x_a, first.x_b, first.x_c):
        assert np.max(analytic_surface_distances(role.points, first.surface)) < 1e-6

    exact = ProceduralSurfaceDataset(
        7,
        num_points=16,
        seed=41,
        split="train",
        training_target="exact_sample",
    )[0]
    independent = ProceduralSurfaceDataset(
        7,
        num_points=16,
        seed=41,
        split="train",
        training_target="independent_resampling",
    )[0]
    assert exact["source_points"].equal(exact["target_points"])
    assert independent["target_points"].equal(independent["independent_points"])
    assert not independent["source_points"].equal(independent["target_points"])


def test_lattice_perturbation_harness_is_deterministic_and_exact() -> None:
    bits = 8
    coordinates = quantize_coordinates(
        torch.tensor([[-0.8, -0.2, 0.4], [0.1, 0.6, 0.9]]), bits
    ).numpy()

    first = perturb_quantized_coordinates(coordinates, bits=bits, bins=2, seed=43)
    repeated = perturb_quantized_coordinates(coordinates, bits=bits, bins=2, seed=43)
    levels = (1 << bits) - 1
    before = np.rint((coordinates + 1.0) * 0.5 * levels)
    after = np.rint((first + 1.0) * 0.5 * levels)

    assert np.array_equal(first, repeated)
    assert np.all(np.abs(after - before) == 2)
    assert np.allclose((first + 1.0) * 0.5 * levels, after, atol=2e-5)


def test_procedural_smoke_runs_both_training_protocols(tmp_path) -> None:
    config = GeometryGateExperimentConfig(
        num_points=8,
        constellation_size=2,
        coordinate_bits=8,
        train_samples=7,
        validation_samples=7,
        parameter_ood_samples=7,
        batch_size=7,
        model_seeds=(47, 53),
        data_seed=59,
        decoder_epochs=1,
        refiner_epochs=1,
        feature_width=8,
        num_heads=2,
        recurrent_steps=1,
        distance_chunk_size=8,
        boundary_band=0.2,
        recall_tolerance=0.2,
        bootstrap_samples=100,
        output_dir=str(tmp_path),
    )

    result = run_geometry_gate_experiment(config, device_name="cpu")
    saved = json.loads((tmp_path / "geometry_gate_metrics.json").read_text())

    assert all(result["contract_checks"].values())
    assert saved["per_cloud_rows"] == 2 * 2 * 2 * 7 * 2 * 3
    assert {row["training_protocol"] for row in saved["training_records"]} == {
        "exact_sample",
        "independent_resampling",
    }
    assert {row["family"] for row in saved["summary"]} == {
        "plane",
        "corner",
        "box",
        "sphere",
        "cylinder",
        "beam",
        "pair",
    }
