# ruff: noqa: E402, I001

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.data import FAMILIES, generate_sample
from pointconstellation.losses import constellation_loss
from pointconstellation.models import (
    ConstellationAutoencoder,
    FarthestPointEncoder,
    FPSAutoencoder,
)
from pointconstellation.quantization import (
    quantization_step,
    quantize_coordinates,
    quantize_ste,
)
from pointconstellation.sweep import SweepSpec, pareto_frontier
from pointconstellation.train import TrainingConfig


def test_procedural_data_are_deterministic_and_normalized() -> None:
    for sample_id, family in enumerate(FAMILIES):
        first = generate_sample(sample_id, num_points=64, seed=11)
        second = generate_sample(sample_id, num_points=64, seed=11)

        assert first.family == family
        np.testing.assert_array_equal(first.points, second.points)
        np.testing.assert_array_equal(first.normals, second.normals)
        assert np.linalg.norm(first.points, axis=1).max() <= 1.000001
        np.testing.assert_allclose(
            np.linalg.norm(first.normals, axis=1), 1.0, atol=1e-5
        )


def test_quantizer_uses_declared_lattice_and_ste_gradient() -> None:
    values = torch.tensor([[[-1.0, -0.2, 0.73]]], requires_grad=True)
    quantized = quantize_ste(values, 8, training=True, jitter=False)
    step = quantization_step(8)
    lattice = (quantized.detach() + 1.0) / step

    assert torch.allclose(lattice, lattice.round(), atol=1e-5)
    quantized.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))
    assert torch.equal(quantize_coordinates(values.detach(), 8), quantized.detach())


def test_autoencoder_contract_and_permutation_invariance() -> None:
    torch.manual_seed(5)
    model = ConstellationAutoencoder(
        num_input_points=32, constellation_size=8, bits=10
    ).eval()
    points = torch.stack(
        [
            torch.from_numpy(generate_sample(index, num_points=32).points)
            for index in range(2)
        ]
    )

    reconstruction, constellation = model(points)
    assert constellation.shape == (2, 8, 3)
    assert reconstruction.shape == (2, 32, 3)

    input_permutation = torch.randperm(32)
    _, permuted_constellation = model(points[:, input_permutation])
    assert torch.allclose(constellation, permuted_constellation, atol=2e-6)

    anchor_permutation = torch.randperm(8)
    permuted_reconstruction = model.decoder(constellation[:, anchor_permutation])
    assert torch.allclose(reconstruction, permuted_reconstruction, atol=2e-6)


def test_fps_encoder_is_parameter_free_quantized_and_permutation_invariant() -> None:
    encoder = FarthestPointEncoder(8, bits=10).eval()
    points = torch.from_numpy(generate_sample(5, num_points=32).points)[None]

    constellation = encoder(points)
    permuted = encoder(points[:, torch.randperm(32)])

    assert constellation.shape == (1, 8, 3)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 0
    assert torch.allclose(constellation, permuted, atol=2e-6)
    step = quantization_step(10)
    lattice = (constellation + 1.0) / step
    assert torch.allclose(lattice, lattice.round(), atol=1e-5)


def test_learned_and_fps_models_start_from_the_same_decoder() -> None:
    torch.manual_seed(17)
    learned = ConstellationAutoencoder(
        num_input_points=32, constellation_size=8, bits=10
    )
    torch.manual_seed(17)
    fps = FPSAutoencoder(num_input_points=32, constellation_size=8, bits=10)

    for name, learned_value in learned.decoder.state_dict().items():
        assert torch.equal(learned_value, fps.decoder.state_dict()[name])


def test_sweep_spec_validates_axes_and_pareto_frontier() -> None:
    with pytest.raises(ValueError, match="between 2 and num_points"):
        SweepSpec(TrainingConfig(num_points=32), (8, 64), (8, 12))

    points = [
        {
            "constellation_size": 4,
            "bits_per_coordinate": 8,
            "coordinate_payload_bits": 96,
            "bits_per_input_point": 0.375,
            "learned": {"chamfer_rmse": 0.5},
        },
        {
            "constellation_size": 4,
            "bits_per_coordinate": 12,
            "coordinate_payload_bits": 144,
            "bits_per_input_point": 0.5625,
            "learned": {"chamfer_rmse": 0.51},
        },
        {
            "constellation_size": 8,
            "bits_per_coordinate": 8,
            "coordinate_payload_bits": 192,
            "bits_per_input_point": 0.75,
            "learned": {"chamfer_rmse": 0.4},
        },
    ]

    frontier = pareto_frontier(points, "learned")
    assert [point["coordinate_payload_bits"] for point in frontier] == [96, 192]


def test_one_training_step_has_finite_loss_and_gradients() -> None:
    torch.manual_seed(9)
    model = ConstellationAutoencoder(num_input_points=32, constellation_size=8, bits=10)
    points = torch.stack(
        [
            torch.from_numpy(generate_sample(index, num_points=32).points)
            for index in range(2)
        ]
    )
    reconstruction, constellation = model(points)
    loss, _ = constellation_loss(reconstruction, points, constellation)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
