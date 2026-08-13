# ruff: noqa: E402

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.data import generate_sample
from pointconstellation.losses import chamfer_squared
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.refiner import CompetitiveConstellationRefiner
from pointconstellation.quantization import quantization_step, quantize_coordinates
from pointconstellation.refiner_experiment import (
    RefinerExperimentConfig,
    run_refiner_experiment,
)


def _points(batch_size: int = 2, num_points: int = 16) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(generate_sample(index, num_points=num_points).points)
            for index in range(batch_size)
        ]
    )


def _same_sets(first: torch.Tensor, second: torch.Tensor, tolerance: float) -> bool:
    distances = ((first[:, :, None] - second[:, None]) ** 2).sum(dim=-1)
    return bool(
        torch.all(distances.amin(dim=2) < tolerance)
        and torch.all(distances.amin(dim=1) < tolerance)
    )


def test_refiner_shapes_responsibilities_and_variable_requests() -> None:
    torch.manual_seed(101)
    model = CompetitiveConstellationRefiner(
        8, bits=9, feature_width=16, num_heads=4, recurrent_steps=2
    ).eval()
    points = _points(num_points=24)

    small, small_history = model(points[:, :12], 4, return_history=True)
    large = model(points, 8)
    point_features = model.point_embedding(points[:, :12])
    slot_features = model.coordinate_embedding(small)
    responsibilities = model.competitive_responsibilities(
        slot_features, small, point_features, points[:, :12]
    )

    assert small.shape == (2, 4, 3)
    assert large.shape == (2, 8, 3)
    assert len(small_history) == 3
    assert responsibilities.shape == (2, 4, 12)
    assert torch.allclose(
        responsibilities.sum(dim=1),
        torch.ones((2, 12)),
        atol=1e-6,
    )


def test_refiner_is_input_invariant_and_slot_equivariant_by_matched_sets() -> None:
    torch.manual_seed(103)
    model = CompetitiveConstellationRefiner(
        6, bits=10, feature_width=16, num_heads=4, recurrent_steps=2
    ).eval()
    points = _points(batch_size=1, num_points=20)

    output = model(points, 6)
    input_permuted = model(points[:, torch.randperm(20)], 6)
    initial = quantize_coordinates(points[:, :6], 10)
    slot_permutation = torch.randperm(6)
    ordered = model(points, 6, initial_constellation=initial)
    slot_permuted = model(
        points,
        6,
        initial_constellation=initial[:, slot_permutation],
    )

    assert _same_sets(output, input_permuted, 1e-9)
    assert _same_sets(ordered, slot_permuted, 1e-9)
    assert torch.allclose(ordered[:, slot_permutation], slot_permuted, atol=1e-6)


def test_every_refinement_state_is_on_the_declared_lattice() -> None:
    torch.manual_seed(107)
    bits = 8
    model = CompetitiveConstellationRefiner(
        4, bits=bits, feature_width=12, num_heads=3, recurrent_steps=3
    ).train()
    _, history = model(_points(batch_size=1), 4, return_history=True)
    step = quantization_step(bits)

    assert len(history) == 4
    for constellation in history:
        lattice = (constellation.detach() + 1.0) / step
        assert torch.allclose(lattice, lattice.round(), atol=2e-5)


def test_unique_projection_is_a_strict_quantized_input_subset() -> None:
    model = CompetitiveConstellationRefiner(
        4, bits=10, feature_width=8, num_heads=2, recurrent_steps=1
    ).eval()
    points = _points(batch_size=1)
    # Deliberately make every free slot prefer the same input point.
    free = points[:, :1].expand(-1, 4, -1).clone()
    projected = model.project_unique_to_input(free, points)
    quantized_input = quantize_coordinates(points, 10)
    membership = ((projected[:, :, None] - quantized_input[:, None]) ** 2).sum(-1)

    assert projected.shape == (1, 4, 3)
    assert torch.all(membership.amin(dim=2) < 1e-10)
    assert len(torch.unique(projected[0], dim=0)) == 4


def test_decoder_gradient_feedback_is_finite_and_decoder_is_immutable() -> None:
    torch.manual_seed(109)
    points = _points(batch_size=2)
    decoder = VariableConstellationDecoder(
        16, 4, feature_width=8, num_heads=2, num_layers=1
    ).eval()
    decoder.requires_grad_(False)
    before = {
        name: value.detach().clone() for name, value in decoder.state_dict().items()
    }
    model = CompetitiveConstellationRefiner(
        4,
        bits=9,
        feature_width=8,
        num_heads=2,
        recurrent_steps=2,
        use_decoder_gradient=True,
    ).train()

    constellation = model(
        points,
        4,
        decoder=decoder,
        target=points,
        num_output_points=16,
    )
    reconstruction = decoder(constellation, num_output_points=16)
    loss = chamfer_squared(reconstruction, points)
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(parameter.grad is None for parameter in decoder.parameters())
    assert all(
        torch.equal(before[name], value) for name, value in decoder.state_dict().items()
    )


def test_tiny_refiner_experiment_runs_end_to_end(tmp_path) -> None:
    config = RefinerExperimentConfig(
        num_points=16,
        input_sizes=(8, 16),
        constellation_sizes=(2, 4),
        bits=8,
        train_samples=7,
        validation_samples=7,
        parameter_ood_samples=7,
        batch_size=7,
        decoder_epochs=1,
        refiner_epochs=1,
        feature_width=8,
        num_heads=2,
        num_layers=1,
        recurrent_steps=1,
        use_decoder_gradient=True,
        output_dir=str(tmp_path),
    )

    result = run_refiner_experiment(config, device_name="cpu")
    saved = json.loads((tmp_path / "metrics.json").read_text())

    assert result["decoder_unchanged"]
    assert saved["decoder_hash_before_refiner"] == saved["decoder_hash_after_refiner"]
    assert (tmp_path / "decoder.pt").exists()
    assert (tmp_path / "refiner.pt").exists()
    assert len(saved["evaluation"]["validation"]) == 4
    assert len(saved["evaluation"]["parameter_ood"]) == 4
    for run in saved["evaluation"]["validation"]:
        assert [point["step"] for point in run["curve"]] == [0, 1]
        for point in run["curve"]:
            assert math_is_finite(point["free"]["chamfer_rmse"])
            assert math_is_finite(point["strict_subset"]["chamfer_rmse"])


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
