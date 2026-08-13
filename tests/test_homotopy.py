# ruff: noqa: E402

from __future__ import annotations

import copy
import json
import math

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.data import generate_sample
from pointconstellation.homotopy_experiment import (
    HomotopyExperimentConfig,
    run_homotopy_experiment,
)
from pointconstellation.models.homotopy import (
    CompressionHomotopyEncoder,
    ConditionalMergeTransition,
    farthest_point_constellation,
)
from pointconstellation.quantization import quantization_step


def _points(batch_size: int = 2, num_points: int = 24) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(
                generate_sample(index, num_points=num_points, seed=43).points
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


def test_homotopy_has_strictly_decreasing_shapes_and_effective_k() -> None:
    torch.manual_seed(201)
    model = CompressionHomotopyEncoder(
        (16, 8, 4), bits=9, feature_width=16, num_heads=4
    ).eval()
    final, history, diagnostics = model(
        _points(), return_history=True, return_diagnostics=True
    )

    assert [stage.shape for stage in history] == [
        (2, 16, 3),
        (2, 8, 3),
        (2, 4, 3),
    ]
    assert final.shape == (2, 4, 3)
    assert len(diagnostics) == 2
    assert [item.hard_assignments.shape for item in diagnostics] == [
        (2, 8, 16),
        (2, 4, 8),
    ]
    assert all(
        1 <= torch.unique(sample, dim=0).shape[0] <= stage.shape[1]
        for stage in history
        for sample in stage
    )
    for item in diagnostics:
        # Every source coordinate has exactly one hard destination.
        assert torch.equal(
            item.hard_assignments.sum(dim=1),
            torch.ones_like(item.hard_assignments[:, 0]),
        )
        # Forced FPS seeds ensure no hard destination is empty.
        assert torch.all(item.hard_assignments.sum(dim=2) >= 1)


def test_input_and_current_anchor_permutations_preserve_matched_sets() -> None:
    torch.manual_seed(203)
    points = _points(batch_size=1)
    model = CompressionHomotopyEncoder(
        (12, 6, 3), bits=10, feature_width=16, num_heads=4
    ).eval()

    original = model(points)
    input_permuted = model(points[:, torch.randperm(points.shape[1])])
    assert _same_sets(original, input_permuted, 1e-9)

    current = farthest_point_constellation(points, 12, 10, training=False)
    transitioned = model.transition(current, 6)
    anchor_permuted = model.transition(current[:, torch.randperm(12)], 6)
    assert _same_sets(transitioned, anchor_permuted, 1e-9)


def test_pruning_merge_relaxation_has_finite_gradients() -> None:
    torch.manual_seed(205)
    current = farthest_point_constellation(
        _points(batch_size=2), 12, 9, training=False
    ).detach()
    current.requires_grad_(True)
    transition = ConditionalMergeTransition(
        bits=9,
        feature_width=16,
        num_heads=4,
        merge_temperature=0.3,
    ).train()
    output, diagnostics = transition(current, 5, return_diagnostics=True)
    loss = output.square().mean()
    loss.backward()

    assert diagnostics.soft_assignments.requires_grad
    assert diagnostics.merge_weights.requires_grad
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    gradients = [
        parameter.grad
        for parameter in transition.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_every_stage_is_exactly_quantized() -> None:
    bits = 8
    model = CompressionHomotopyEncoder(
        (12, 6, 3), bits=bits, feature_width=12, num_heads=3
    ).train()
    _, history, _ = model(_points(batch_size=1), return_history=True)
    step = quantization_step(bits)

    for constellation in history:
        lattice = (constellation.detach() + 1.0) / step
        assert torch.allclose(lattice, lattice.round(), atol=2e-5)


def test_direct_and_homotopy_can_use_bitwise_matched_initialization() -> None:
    torch.manual_seed(207)
    base = CompressionHomotopyEncoder(
        (12, 6, 3), bits=9, feature_width=16, num_heads=4
    ).eval()
    homotopy = copy.deepcopy(base)
    direct = copy.deepcopy(base)

    assert all(
        torch.equal(homotopy.state_dict()[name], value)
        for name, value in direct.state_dict().items()
    )
    points = _points(batch_size=1)
    homotopy_first = homotopy(points, stage_sizes=(12, 6))
    direct_first = direct(points, stage_sizes=(12, 6))
    assert torch.equal(homotopy_first, direct_first)


def test_homotopy_config_rejects_non_decreasing_paths() -> None:
    with pytest.raises(ValueError, match="strictly decreasing"):
        HomotopyExperimentConfig(stage_sizes=(16, 8, 8))
    with pytest.raises(ValueError, match="dense stage"):
        HomotopyExperimentConfig(input_size=16, stage_sizes=(24, 8))


def test_tiny_homotopy_experiment_runs_with_frozen_decoder(tmp_path) -> None:
    config = HomotopyExperimentConfig(
        num_points=16,
        input_size=16,
        stage_sizes=(8, 4, 2),
        bits=8,
        train_samples=3,
        validation_samples=2,
        parameter_ood_samples=2,
        batch_size=3,
        decoder_epochs=1,
        encoder_epochs_per_stage=1,
        feature_width=8,
        num_heads=2,
        output_dir=str(tmp_path),
    )
    result = run_homotopy_experiment(config, device_name="cpu")
    saved = json.loads((tmp_path / "metrics.json").read_text())

    assert result["decoder_unchanged"]
    assert saved["decoder_hash_before_encoders"] == saved["decoder_hash_final"]
    assert saved["initial_parameters_matched"]
    assert saved["update_budget_matched"]
    assert saved["homotopy_updates"] == saved["direct_updates"] == 2
    assert (tmp_path / "decoder.pt").exists()
    assert (tmp_path / "homotopy_encoder.pt").exists()
    assert (tmp_path / "direct_encoder.pt").exists()

    for split in ("validation", "parameter_ood"):
        evaluation = saved["evaluation"][split]
        for method in ("homotopy", "direct"):
            assert [stage["requested_k"] for stage in evaluation[method]] == [8, 4, 2]
            for stage in evaluation[method]:
                assert stage["exact_quantized"]
                assert 1 <= stage["effective_k_mean"] <= stage["requested_k"]
                assert math.isfinite(stage["chamfer_rmse"])
                assert math.isfinite(stage["coverage_rmse"])
                assert math.isfinite(stage["surface_proxy_rmse"])
        assert math.isfinite(
            evaluation["target_k_gap"]["homotopy_minus_direct_chamfer_rmse"]
        )
