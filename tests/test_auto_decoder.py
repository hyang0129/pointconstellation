# ruff: noqa: E402, I001

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.auto_decoder_experiment import (
    AutoDecoderConfig,
    infer_heldout_coordinates,
    run_auto_decoder_experiment,
)
from pointconstellation.data import generate_sample
from pointconstellation.losses import chamfer_squared, pairwise_squared
from pointconstellation.models.coordinate_auto_decoder import (
    CoordinateAutoDecoder,
    CoordinateOnlyDecoder,
    LiteralCoordinateBank,
    PermutationInvariantAmortizer,
    project_to_input_surface,
    quantize_constellation,
)
from pointconstellation.quantization import quantization_step, quantize_coordinates


def _clouds(count: int = 3, num_points: int = 16, split: str = "train"):
    return torch.stack(
        [
            torch.from_numpy(
                generate_sample(
                    index,
                    num_points=num_points,
                    seed=19,
                    split=split,
                ).points
            )
            for index in range(count)
        ]
    )


def test_literal_coordinate_only_contract() -> None:
    points = _clouds()
    model = CoordinateAutoDecoder(
        num_clouds=3,
        num_restarts=2,
        num_output_points=16,
        constellation_size=4,
        bits=8,
        initial_clouds=points,
        feature_width=16,
    )
    reconstruction, codes = model(torch.arange(3), straight_through=False)

    assert model.codes.coordinates.shape == (3, 2, 4, 3)
    assert model.codes.coordinates.numel() == 3 * 2 * 4 * 3
    assert codes.shape == (3, 2, 4, 3)
    assert reconstruction.shape == (3, 2, 16, 3)
    # The decoder API cannot receive a cloud ID, input cloud, or feature tensor.
    assert model.decoder(codes[:, 0]).shape == (3, 16, 3)


def test_exact_and_ste_coordinate_quantization() -> None:
    values = torch.tensor([[[-0.73, 0.11, 0.92]]], requires_grad=True)
    exact = quantize_constellation(values, 8, straight_through=False)
    estimated = quantize_constellation(values, 8, straight_through=True)
    step = quantization_step(8)

    assert torch.equal(exact, estimated.detach())
    assert torch.allclose((exact + 1.0) / step, ((exact + 1.0) / step).round())
    estimated.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_projected_mode_selects_quantized_input_surface_points() -> None:
    points = _clouds(count=2)
    bank = LiteralCoordinateBank(
        2,
        2,
        4,
        bits=8,
        mode="projected",
        initial_clouds=points,
    )
    coordinates = bank(
        torch.arange(2),
        reference_points=points,
        straight_through=False,
    )
    quantized_points = quantize_coordinates(points, 8)
    references = quantized_points[:, None].expand(-1, 2, -1, -1).flatten(0, 1)
    distances = pairwise_squared(coordinates.flatten(0, 1), references)

    assert torch.all(distances.amin(dim=-1) < 1e-6)
    with pytest.raises(ValueError, match="reference_points"):
        bank(torch.arange(2))


def test_projection_ste_has_finite_identity_gradient() -> None:
    points = _clouds(count=1)
    values = torch.zeros((1, 2, 3), requires_grad=True)
    projected = project_to_input_surface(values, points, straight_through=True)
    projected.square().sum().backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_coordinate_restarts_are_distinct() -> None:
    bank = LiteralCoordinateBank(
        3,
        3,
        4,
        bits=8,
        initial_clouds=_clouds(),
        seed=31,
    )
    differences = (
        (bank.coordinates[:, 1:] - bank.coordinates[:, :1]).abs().sum(dim=(2, 3))
    )
    assert torch.all(differences > 0)


def test_coordinate_and_decoder_gradients_are_finite() -> None:
    points = _clouds(count=2)
    model = CoordinateAutoDecoder(
        num_clouds=2,
        num_restarts=2,
        num_output_points=16,
        constellation_size=4,
        bits=8,
        initial_clouds=points,
        feature_width=16,
    )
    reconstruction, _ = model(torch.arange(2), straight_through=True)
    target = points[:, None].expand_as(reconstruction)
    loss = chamfer_squared(reconstruction.flatten(0, 1), target.flatten(0, 1))
    loss.backward()

    assert model.codes.coordinates.grad is not None
    assert torch.isfinite(model.codes.coordinates.grad).all()
    gradients = [
        parameter.grad
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_heldout_refinement_keeps_decoder_frozen() -> None:
    points = _clouds(count=2, split="validation")
    decoder = CoordinateOnlyDecoder(16, feature_width=16)
    amortizer = PermutationInvariantAmortizer(4, feature_width=16)
    before = {
        name: value.detach().clone() for name, value in decoder.state_dict().items()
    }
    config = replace(
        AutoDecoderConfig(),
        num_points=16,
        constellation_size=4,
        train_samples=2,
        heldout_samples=2,
        code_restarts=2,
        heldout_restarts=3,
        heldout_steps=2,
        feature_width=16,
    )

    inferred = infer_heldout_coordinates(
        decoder,
        amortizer,
        points,
        config,
        mode="unrestricted",
    )

    assert inferred.decoder_unchanged
    assert inferred.one_shot_coordinates.shape == (2, 4, 3)
    assert inferred.refined_coordinates.shape == (2, 4, 3)
    assert torch.all(inferred.refined_losses <= inferred.one_shot_losses + 1e-8)
    for name, value in decoder.state_dict().items():
        assert torch.equal(value, before[name])


def test_tiny_end_to_end_experiment_writes_metrics(tmp_path) -> None:
    config = AutoDecoderConfig(
        num_points=16,
        constellation_size=4,
        bits=8,
        train_samples=3,
        heldout_samples=2,
        code_restarts=2,
        heldout_restarts=2,
        alternating_epochs=1,
        coordinate_updates_per_epoch=1,
        decoder_updates_per_epoch=1,
        amortizer_epochs=1,
        heldout_steps=1,
        noise_scales=(0.0, 0.02),
        modes=("unrestricted",),
        feature_width=12,
        output_dir=str(tmp_path),
    )
    result = run_auto_decoder_experiment(config, device_name="cpu")
    run = result["runs"]["unrestricted"]

    assert (tmp_path / "metrics.json").exists()
    assert run["clean"]["chamfer_rmse"] >= 0
    assert run["noisy"]["mean_nonzero_chamfer_rmse"] >= 0
    assert run["heldout"]["decoder_unchanged"]
    assert run["heldout"]["refined_mean_chamfer"] <= (
        run["heldout"]["one_shot_mean_chamfer"] + 1e-8
    )
    assert run["behavior"]["literal_code_shape"] == [3, 2, 4, 3]
