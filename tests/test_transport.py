"""Focused tests for Experiment 008 mass-aware objectives."""

# ruff: noqa: E402

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.data import generate_sample
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.refiner import CompetitiveConstellationRefiner
from pointconstellation.models.transport import (
    BalancedResponsibilityRefiner,
    balanced_anchor_responsibilities,
    balanced_transport_squared,
    density_aware_chamfer_squared,
    sinkhorn_transport_plan,
)
from pointconstellation.quantization import quantization_step
from pointconstellation.transport_experiment import (
    TransportExperimentConfig,
    run_transport_experiment,
)


def _points(batch_size: int, num_points: int, *, split: str = "train"):
    return torch.stack(
        [
            torch.from_numpy(
                generate_sample(
                    index,
                    num_points=num_points,
                    seed=23,
                    split=split,
                ).points
            )
            for index in range(batch_size)
        ]
    )


def test_sinkhorn_marginals_and_finite_gradients() -> None:
    torch.manual_seed(3)
    first = torch.randn(2, 4, 3, requires_grad=True)
    second = torch.randn(2, 7, 3, requires_grad=True)
    plan = sinkhorn_transport_plan(first, second, epsilon=0.2, iterations=100)

    assert plan.shape == (2, 4, 7)
    assert torch.allclose(plan.sum(dim=2), torch.full((2, 4), 1 / 4), atol=2e-4)
    assert torch.allclose(plan.sum(dim=1), torch.full((2, 7), 1 / 7), atol=2e-4)
    loss = balanced_transport_squared(first, second, epsilon=0.2, iterations=100)
    loss.backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


def test_balanced_anchor_responsibilities_have_equal_mass() -> None:
    anchors = _points(2, 8)[:, :4]
    points = _points(2, 12)
    responsibilities = balanced_anchor_responsibilities(
        anchors, points, epsilon=0.1, iterations=100
    )

    assert torch.allclose(responsibilities.sum(dim=1), torch.ones(2, 12), atol=2e-4)
    assert torch.allclose(
        responsibilities.sum(dim=2), torch.full((2, 4), 3.0), atol=2e-4
    )


def test_balanced_refiner_uses_equal_mass_internal_responsibilities() -> None:
    points = _points(2, 12)
    refiner = BalancedResponsibilityRefiner(
        4,
        bits=8,
        feature_width=16,
        num_heads=4,
        recurrent_steps=1,
        sinkhorn_epsilon=0.1,
        sinkhorn_iterations=100,
    )
    coordinates = points[:, :4]
    slot_features = refiner.coordinate_embedding(coordinates)
    point_features = refiner.point_embedding(points)
    responsibilities = refiner.competitive_responsibilities(
        slot_features, coordinates, point_features, points
    )

    assert torch.allclose(responsibilities.sum(dim=1), torch.ones(2, 12), atol=2e-4)
    assert torch.allclose(
        responsibilities.sum(dim=2), torch.full((2, 4), 3.0), atol=2e-4
    )


@pytest.mark.parametrize(
    "loss_function",
    (balanced_transport_squared, density_aware_chamfer_squared),
)
def test_transport_losses_are_permutation_invariant(loss_function) -> None:
    first = _points(2, 8)
    second = _points(2, 11, split="validation")
    expected = loss_function(first, second)
    permuted = loss_function(first[:, torch.randperm(8)], second[:, torch.randperm(11)])

    assert torch.allclose(expected, permuted, atol=2e-6)


def test_density_loss_penalizes_duplicate_collapse() -> None:
    target = torch.tensor(
        [[[-0.9, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.9, 0.0, 0.0]]]
    )
    spread = target.clone()
    duplicated = torch.tensor(
        [[[-0.9, 0.0, 0.0], [-0.9, 0.0, 0.0], [0.3, 0.0, 0.0], [0.9, 0.0, 0.0]]]
    )

    spread_loss = density_aware_chamfer_squared(spread, target, temperature=0.02)
    duplicate_loss = density_aware_chamfer_squared(duplicated, target, temperature=0.02)

    assert spread_loss.item() < 1e-5
    assert duplicate_loss.item() > spread_loss.item() + 0.05


def test_refiner_message_is_coordinate_only_and_exactly_quantized() -> None:
    points = _points(2, 12)
    decoder = VariableConstellationDecoder(
        12, 4, feature_width=16, num_heads=4, num_layers=1
    ).eval()
    decoder.requires_grad_(False)
    refiner = CompetitiveConstellationRefiner(
        4,
        bits=8,
        feature_width=16,
        num_heads=4,
        recurrent_steps=2,
    ).eval()
    constellation = refiner(
        points,
        4,
        decoder=decoder,
        target=points,
        num_output_points=12,
    )
    step = quantization_step(8)
    lattice = (constellation + 1.0) / step

    assert isinstance(constellation, torch.Tensor)
    assert constellation.shape == (2, 4, 3)
    assert torch.allclose(lattice, lattice.round(), atol=1e-5)


def test_tiny_end_to_end_is_matched_and_decoder_frozen(tmp_path) -> None:
    config = TransportExperimentConfig(
        num_points=8,
        input_size=8,
        constellation_size=2,
        bits=8,
        train_samples=3,
        validation_samples=2,
        parameter_ood_samples=2,
        batch_size=1,
        decoder_epochs=1,
        refiner_epochs=1,
        feature_width=8,
        num_heads=2,
        num_layers=1,
        recurrent_steps=1,
        sinkhorn_iterations=8,
        output_dir=str(tmp_path),
    )
    result = run_transport_experiment(config, device_name="cpu")
    saved = json.loads((tmp_path / "metrics.json").read_text())

    assert result["matched_initialization"]
    assert len(set(result["refiner_initial_hashes"].values())) == 1
    assert result["matched_data_order"]
    assert len(set(result["training_order_hashes"].values())) == 1
    assert result["decoder_unchanged"]
    assert (
        result["decoder_hash_before_refiners"]
        == (result["decoder_hash_after_refiners"])
    )
    assert set(result["arms"]) == {
        "chamfer",
        "density_aware",
        "balanced_transport",
    }
    for objective, arm in result["arms"].items():
        assert (tmp_path / f"refiner_{objective}.pt").exists()
        assert len(arm["validation_curve"]) == 2
        assert len(arm["parameter_ood_curve"]) == 2
        assert arm["validation_curve"][-1]["reconstruction_chamfer_rmse"] >= 0
        assert arm["validation_curve"][-1]["coverage_rmse"] >= 0
        assert arm["validation_curve"][-1]["anchor_mass_imbalance"] >= 0
        assert arm["validation_curve"][-1]["maximum_lattice_error"] < 1e-4
    assert (tmp_path / "decoder.pt").exists()
    assert saved["message_contract"]["coordinate_only"]
    assert saved["message_contract"]["exact_final_quantization"]
