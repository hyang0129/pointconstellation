# ruff: noqa: E402

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.crossplay_experiment import (
    CrossplayExperimentConfig,
    run_crossplay_experiment,
)
from pointconstellation.data import generate_sample
from pointconstellation.losses import chamfer_squared
from pointconstellation.models.crossplay import (
    aggregate_decoder_losses,
    make_crossplay_refiner,
    make_decoder_population,
    reconstruction_loss_matrix,
)
from pointconstellation.quantization import quantization_step


def _points(batch_size: int = 2, num_points: int = 16) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(generate_sample(index, num_points=num_points).points)
            for index in range(batch_size)
        ]
    )


def _population(size: int = 2):
    return make_decoder_population(
        size,
        max_output_points=16,
        max_constellation_size=4,
        feature_width=8,
        num_heads=2,
        num_layers=1,
        seed=211,
    )


def _refiner():
    return make_crossplay_refiner(
        4,
        bits=8,
        feature_width=8,
        num_heads=2,
        recurrent_steps=1,
        responsibility_temperature=0.2,
        maximum_update=0.1,
    )


def test_population_members_have_independent_reproducible_initialization() -> None:
    first = _population(3)
    repeated = _population(3)

    first_parameters = [next(decoder.parameters()) for decoder in first.decoders]
    repeated_parameters = [next(decoder.parameters()) for decoder in repeated.decoders]
    assert len({id(decoder) for decoder in first.decoders}) == 3
    assert not torch.equal(first_parameters[0], first_parameters[1])
    assert not torch.equal(first_parameters[1], first_parameters[2])
    assert all(
        torch.equal(left, right)
        for left, right in zip(first_parameters, repeated_parameters, strict=True)
    )


def test_aggregate_loss_is_mean_plus_weighted_worst() -> None:
    losses = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    aggregate = aggregate_decoder_losses(losses, worst_weight=0.25)
    aggregate.backward()

    assert torch.allclose(aggregate, torch.tensor(10.0 / 3.0))
    assert torch.allclose(losses.grad, torch.tensor([1 / 3, 1 / 3, 7 / 12]))


def test_crossplay_matrix_shape_and_values_match_direct_losses() -> None:
    torch.manual_seed(223)
    target = _points()
    population = _population()
    messages = (target[:, :2], target[:, 2:4])
    matrix = reconstruction_loss_matrix(
        messages, population, target, num_output_points=16
    )

    assert matrix.shape == (2, 2)
    expected = chamfer_squared(
        population.decoders[1](messages[0], num_output_points=16), target
    )
    assert torch.allclose(matrix[0, 1], expected)


def test_population_objective_has_finite_encoder_gradients_and_frozen_decoders() -> (
    None
):
    torch.manual_seed(227)
    target = _points()
    source = target[:, :8]
    population = _population()
    population.eval().requires_grad_(False)
    before = {
        f"{index}.{name}": value.detach().clone()
        for index, decoder in enumerate(population.decoders)
        for name, value in decoder.state_dict().items()
    }
    refiner = _refiner().train()

    message = refiner(source, 4)
    losses = torch.stack(
        [
            chamfer_squared(reconstruction, target)
            for reconstruction in population.forward_all(message, num_output_points=16)
        ]
    )
    aggregate_decoder_losses(losses, worst_weight=0.2).backward()

    gradients = [
        parameter.grad
        for parameter in refiner.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(
        parameter.grad is None
        for decoder in population.decoders
        for parameter in decoder.parameters()
    )
    assert all(
        torch.equal(before[f"{index}.{name}"], value)
        for index, decoder in enumerate(population.decoders)
        for name, value in decoder.state_dict().items()
    )


def test_refiner_message_is_exactly_quantized_coordinates_only() -> None:
    refiner = _refiner().train()
    message = refiner(_points()[:, :8], 4)
    step = quantization_step(8)
    lattice = (message.detach() + 1.0) / step

    assert message.shape == (2, 4, 3)
    assert torch.allclose(lattice, lattice.round(), atol=2e-5)


def test_crossplay_config_validation() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        CrossplayExperimentConfig(population_size=1)
    with pytest.raises(ValueError, match="cannot be negative"):
        CrossplayExperimentConfig(worst_decoder_weight=-0.1)
    with pytest.raises(ValueError, match="between zero and one"):
        CrossplayExperimentConfig(fps_probability=1.1)


def test_tiny_crossplay_experiment_runs_end_to_end(tmp_path) -> None:
    config = CrossplayExperimentConfig(
        num_points=16,
        input_sizes=(8, 16),
        constellation_sizes=(2, 4),
        bits=8,
        train_samples=7,
        validation_samples=7,
        parameter_ood_samples=7,
        batch_size=7,
        population_size=2,
        decoder_epochs=1,
        refiner_epochs=1,
        feature_width=8,
        num_heads=2,
        num_layers=1,
        recurrent_steps=1,
        output_dir=str(tmp_path),
    )

    result = run_crossplay_experiment(config, device_name="cpu")
    saved = json.loads((tmp_path / "metrics.json").read_text())

    assert result["decoder_population_initialized_independently"]
    assert result["decoders_unchanged_during_refiner_training"]
    assert result["refiner_initialization_matched"]
    assert (
        saved["decoder_hashes_before_refiners"]
        == saved["decoder_hashes_after_refiners"]
    )
    assert (tmp_path / "decoder_0.pt").exists()
    assert (tmp_path / "decoder_1.pt").exists()
    assert (tmp_path / "matched_refiner.pt").exists()
    assert (tmp_path / "population_refiner.pt").exists()
    assert len(saved["evaluation"]["validation"]) == 4
    assert len(saved["evaluation"]["parameter_ood"]) == 4
    for run in saved["evaluation"]["validation"]:
        matrix = run["crossplay_chamfer_rmse_matrix"]
        assert len(matrix) == 2
        assert all(len(row) == 2 for row in matrix)
        assert all(math.isfinite(value) for row in matrix for value in row)
        assert set(run["message_resampling_stability_rmse"]) == {
            "matched_single_decoder",
            "population",
        }
        assert all(
            error < 2e-5 for error in run["maximum_quantization_lattice_error"].values()
        )
