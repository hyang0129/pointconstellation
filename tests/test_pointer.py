# ruff: noqa: E402

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pointconstellation.data import generate_sample
from pointconstellation.losses import chamfer_squared
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.pointer import AutoregressivePointerSubsetEncoder
from pointconstellation.pointer_experiment import (
    PointerExperimentConfig,
    run_pointer_experiment,
)
from pointconstellation.quantization import quantization_step, quantize_coordinates


def _points(batch_size: int = 2, num_points: int = 20) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(
                generate_sample(index, num_points=num_points, seed=53).points
            )
            for index in range(batch_size)
        ]
    )


def _same_sets(first: torch.Tensor, second: torch.Tensor, tolerance: float) -> bool:
    distances = ((first[:, :, None] - second[:, None]) ** 2).sum(dim=-1)
    return bool(
        torch.all(distances.amin(dim=2) < tolerance)
        and torch.all(distances.amin(dim=1) < tolerance)
    )


def test_pointer_selects_unique_members_of_quantized_input() -> None:
    torch.manual_seed(301)
    points = _points()
    model = AutoregressivePointerSubsetEncoder(
        8, bits=9, feature_width=16, num_heads=4
    ).eval()
    output, trace = model(points, 6, return_trace=True)
    quantized_input = quantize_coordinates(points, 9)
    membership = ((output[:, :, None] - quantized_input[:, None]) ** 2).sum(-1)

    assert output.shape == (2, 6, 3)
    assert trace.indices.shape == (2, 6)
    assert torch.all(membership.amin(dim=2) < 1e-10)
    assert all(len(torch.unique(row)) == 6 for row in trace.indices)


def test_next_logits_change_with_selected_set_and_residual() -> None:
    torch.manual_seed(303)
    points = _points(batch_size=1)
    model = AutoregressivePointerSubsetEncoder(
        8, bits=10, feature_width=16, num_heads=4
    ).eval()
    empty = torch.empty((1, 0), dtype=torch.long)
    first = model.conditional_logits(points, empty, 6)
    after_zero = model.conditional_logits(points, torch.tensor([[0]]), 6)
    after_one = model.conditional_logits(points, torch.tensor([[1]]), 6)

    remaining = torch.arange(points.shape[1]) >= 2
    assert not torch.allclose(first[:, remaining], after_zero[:, remaining])
    assert not torch.allclose(after_zero[:, remaining], after_one[:, remaining])
    assert after_zero[0, 0] == torch.finfo(after_zero.dtype).min
    assert after_one[0, 1] == torch.finfo(after_one.dtype).min


def test_pointer_is_input_permutation_invariant_by_matched_sets() -> None:
    torch.manual_seed(305)
    points = _points(batch_size=1)
    model = AutoregressivePointerSubsetEncoder(
        8, bits=10, feature_width=16, num_heads=4
    ).eval()
    original = model(points, 6)
    permuted = model(points[:, torch.randperm(points.shape[1])], 6)

    assert _same_sets(original, permuted, 1e-9)


def test_pointer_accepts_variable_input_and_requested_sizes() -> None:
    model = AutoregressivePointerSubsetEncoder(
        8, bits=9, feature_width=12, num_heads=3
    ).eval()
    points = _points()

    assert model(points[:, :12], 4).shape == (2, 4, 3)
    assert model(points, 8).shape == (2, 8, 3)
    with pytest.raises(ValueError, match="input point count"):
        model(points[:, :4], 8)


def test_straight_through_pointer_has_finite_gradients() -> None:
    torch.manual_seed(307)
    model = AutoregressivePointerSubsetEncoder(
        6,
        bits=9,
        feature_width=16,
        num_heads=4,
        stochastic_training=True,
    ).train()
    output, trace = model(_points(), 4, stochastic=True, return_trace=True)
    loss = output.square().mean() - 0.001 * trace.entropies.mean()
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert model.conditional_score[-1].weight.grad is not None
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_hard_evaluation_is_exactly_quantized() -> None:
    bits = 8
    model = AutoregressivePointerSubsetEncoder(
        6, bits=bits, feature_width=12, num_heads=3
    ).eval()
    output = model(_points(), 6)
    step = quantization_step(bits)
    lattice = (output + 1.0) / step

    assert torch.allclose(lattice, lattice.round(), atol=2e-5)


class _IdentitySetDecoder(nn.Module):
    def forward(
        self, constellation: torch.Tensor, *, num_output_points: int | None = None
    ) -> torch.Tensor:
        del num_output_points
        return constellation


def test_beam_search_selects_a_lower_loss_completed_candidate() -> None:
    torch.manual_seed(309)
    points = torch.tensor(
        [
            [
                [-0.9, -0.8, 0.0],
                [-0.3, 0.7, 0.0],
                [0.4, -0.5, 0.0],
                [0.9, 0.8, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    model = AutoregressivePointerSubsetEncoder(
        2, bits=10, feature_width=8, num_heads=2
    ).eval()
    greedy_output = model(points, 2, return_trace=True)
    assert isinstance(greedy_output, tuple)
    greedy_indices = set(greedy_output[1].indices[0].tolist())
    target_indices = sorted(set(range(4)) - greedy_indices)
    target = quantize_coordinates(points[:, target_indices], 10)
    decoder = _IdentitySetDecoder().eval().requires_grad_(False)

    result = model.beam_search(
        points,
        2,
        decoder=decoder,
        target=target,
        num_output_points=2,
        beam_width=12,
        branch_factor=4,
    )

    assert result.losses.item() < result.greedy_losses.item()
    assert set(result.indices[0].tolist()) == set(target_indices)
    assert result.decoder_evaluations.item() > 1


def test_decoder_is_frozen_during_pointer_gradient() -> None:
    torch.manual_seed(311)
    points = _points(batch_size=2, num_points=16)
    decoder = VariableConstellationDecoder(
        16, 6, feature_width=8, num_heads=2, num_layers=1
    ).eval()
    decoder.requires_grad_(False)
    before = {
        name: value.detach().clone() for name, value in decoder.state_dict().items()
    }
    pointer = AutoregressivePointerSubsetEncoder(
        6, bits=8, feature_width=8, num_heads=2
    ).train()
    constellation = pointer(points, 4, stochastic=False)
    reconstruction = decoder(constellation, num_output_points=16)
    chamfer_squared(reconstruction, points).backward()

    assert any(parameter.grad is not None for parameter in pointer.parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())
    assert all(
        torch.equal(before[name], value) for name, value in decoder.state_dict().items()
    )


def test_tiny_pointer_experiment_runs_end_to_end(tmp_path) -> None:
    config = PointerExperimentConfig(
        num_points=16,
        input_sizes=(8, 16),
        constellation_sizes=(2, 4),
        bits=8,
        train_samples=3,
        validation_samples=2,
        parameter_ood_samples=2,
        batch_size=3,
        decoder_epochs=1,
        selector_epochs=1,
        feature_width=8,
        num_heads=2,
        beam_width=2,
        beam_branch_factor=2,
        output_dir=str(tmp_path),
    )
    result = run_pointer_experiment(config, device_name="cpu")
    saved = json.loads((tmp_path / "metrics.json").read_text())

    assert result["decoder_unchanged"]
    assert saved["decoder_hash_before_selectors"] == saved["decoder_hash_final"]
    assert saved["common_initialization_matched"]
    assert saved["update_budget_matched"]
    assert saved["pointer_updates"] == saved["scalar_updates"] == 4
    assert len(saved["training_operating_points"]) == 4
    assert saved["pointer_scalar_stochasticity_matched"]
    assert saved["pointer_scalar_quantization_jitter_matched"]
    assert (tmp_path / "decoder.pt").exists()
    assert (tmp_path / "pointer.pt").exists()
    assert (tmp_path / "scalar.pt").exists()
    assert len(saved["evaluation"]["validation"]) == 4
    assert len(saved["evaluation"]["parameter_ood"]) == 4

    for run in saved["evaluation"]["validation"]:
        methods = run["methods"]
        for method in ("pointer_greedy", "pointer_beam", "scalar", "fps"):
            assert methods[method]["fully_unique_fraction"] == 1.0
            assert methods[method]["exact_quantized"]
            assert math.isfinite(methods[method]["chamfer_rmse"])
            assert math.isfinite(methods[method]["coverage_rmse"])
        assert methods["beam_gain"]["greedy_minus_beam_chamfer_rmse"] >= -1e-8
