# ruff: noqa: E402, I001

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.data import FAMILIES, generate_sample
from pointconstellation.encoder_isolation import (
    EncoderIsolationSpec,
    encoder_isolation_gate,
)
from pointconstellation.feature_bitstream import decode_features, encode_features
from pointconstellation.feature_codec_benchmark import (
    FeatureCodecBenchmarkConfig,
    run_feature_codec_benchmark,
)
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    run_stability_experiment,
)
from pointconstellation.bottleneck_audit import (
    BottleneckAuditConfig,
    bottleneck_audit_gate,
)
from pointconstellation.losses import constellation_loss
from pointconstellation.models import (
    ConstellationAutoencoder,
    FarthestPointEncoder,
    FPSAutoencoder,
    HardSubsetConstellationEncoder,
    ProgressiveSubsetEncoder,
    RelationAwareConstellationAutoencoder,
    RelationAwareFPSAutoencoder,
    RelationAwareSubsetAutoencoder,
    VariableConstellationDecoder,
)
from pointconstellation.quantization import (
    quantization_step,
    quantize_coordinates,
    quantize_ste,
)
from pointconstellation.models.feature_codec import VariableFeatureCodec
from pointconstellation.selected_rate import SelectedRateSpec, rate_curve_gate
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


def test_relation_model_contract_and_set_permutation_invariance() -> None:
    torch.manual_seed(23)
    model = RelationAwareConstellationAutoencoder(
        num_input_points=32, constellation_size=8, bits=10
    ).eval()
    points = torch.stack(
        [
            torch.from_numpy(generate_sample(index, num_points=32).points)
            for index in range(2)
        ]
    )

    reconstruction, constellation = model(points)
    _, input_permuted_constellation = model(points[:, torch.randperm(32)])
    anchor_permuted_reconstruction = model.decoder(constellation[:, torch.randperm(8)])

    assert reconstruction.shape == (2, 32, 3)
    assert constellation.shape == (2, 8, 3)
    assert torch.allclose(constellation, input_permuted_constellation, atol=2e-6)
    assert torch.allclose(reconstruction, anchor_permuted_reconstruction, atol=2e-6)


def test_relation_learned_and_fps_start_from_the_same_decoder() -> None:
    torch.manual_seed(29)
    learned = RelationAwareConstellationAutoencoder(
        num_input_points=32, constellation_size=8, bits=10
    )
    torch.manual_seed(29)
    fps = RelationAwareFPSAutoencoder(
        num_input_points=32, constellation_size=8, bits=10
    )

    for name, learned_value in learned.decoder.state_dict().items():
        assert torch.equal(learned_value, fps.decoder.state_dict()[name])


def test_hard_subset_encoder_selects_unique_quantized_input_points() -> None:
    torch.manual_seed(31)
    encoder = HardSubsetConstellationEncoder(8, bits=10).eval()
    points = torch.from_numpy(generate_sample(4, num_points=32).points)[None]

    constellation = encoder(points)
    quantized_input = quantize_coordinates(points, 10)
    distances = ((constellation[:, :, None] - quantized_input[:, None]) ** 2).sum(
        dim=-1
    )
    permuted = encoder(points[:, torch.randperm(32)])

    assert torch.all(distances.amin(dim=-1) < 1e-10)
    assert len(torch.unique(constellation[0], dim=0)) == 8
    assert torch.allclose(constellation, permuted, atol=2e-6)


def test_hard_subset_straight_through_selection_has_finite_gradients() -> None:
    torch.manual_seed(37)
    encoder = HardSubsetConstellationEncoder(4, bits=10).train()
    points = torch.from_numpy(generate_sample(2, num_points=16).points)[None]

    constellation = encoder(points)
    constellation.square().sum().backward()

    assert encoder.selection_queries.weight.grad is not None
    assert torch.isfinite(encoder.selection_queries.weight.grad).all()


def test_relation_subset_and_fps_start_from_the_same_decoder() -> None:
    torch.manual_seed(41)
    subset = RelationAwareSubsetAutoencoder(
        num_input_points=32, constellation_size=8, bits=10
    )
    torch.manual_seed(41)
    fps = RelationAwareFPSAutoencoder(
        num_input_points=32, constellation_size=8, bits=10
    )

    for name, subset_value in subset.decoder.state_dict().items():
        assert torch.equal(subset_value, fps.decoder.state_dict()[name])


def test_progressive_subset_encoder_is_nested_quantized_and_differentiable() -> None:
    torch.manual_seed(43)
    encoder = ProgressiveSubsetEncoder(8, bits=10).eval()
    points = torch.from_numpy(generate_sample(3, num_points=32).points)[None]

    small = encoder(points, 4)
    large = encoder(points, 8)
    quantized_input = quantize_coordinates(points, 10)
    membership = ((small[:, :, None] - large[:, None]) ** 2).sum(dim=-1)
    input_membership = ((large[:, :, None] - quantized_input[:, None]) ** 2).sum(dim=-1)
    permuted = encoder(points[:, torch.randperm(32)], 8)
    permutation_membership = ((large[:, :, None] - permuted[:, None]) ** 2).sum(dim=-1)

    assert torch.all(membership.amin(dim=-1) < 1e-10)
    assert torch.all(input_membership.amin(dim=-1) < 1e-10)
    assert len(torch.unique(large[0], dim=0)) == 8
    assert torch.all(permutation_membership.amin(dim=-1) < 1e-10)

    encoder.train()
    constellation = encoder(points, 4)
    constellation.square().sum().backward()
    assert encoder.score_head[-1].weight.grad is not None
    assert torch.isfinite(encoder.score_head[-1].weight.grad).all()


def test_variable_decoder_preserves_anchors_and_accepts_variable_sizes() -> None:
    torch.manual_seed(47)
    decoder = VariableConstellationDecoder(32, 8).eval()
    points = torch.from_numpy(generate_sample(1, num_points=32).points)[None]
    anchors = quantize_coordinates(points[:, :8], 10)

    small = decoder(anchors[:, :4], num_output_points=16)
    large = decoder(anchors, num_output_points=24)
    permuted_anchors = anchors[:, torch.randperm(8)]
    permuted = decoder(permuted_anchors, num_output_points=24)

    assert small.shape == (1, 16, 3)
    assert large.shape == (1, 24, 3)
    assert torch.equal(small[:, :4], anchors[:, :4])
    assert torch.equal(large[:, :8], anchors)
    assert torch.equal(permuted[:, :8], permuted_anchors)
    assert torch.allclose(large[:, 8:], permuted[:, 8:], atol=2e-6)


def test_feature_codec_contract_and_bitstream_round_trip() -> None:
    torch.manual_seed(53)
    codec = VariableFeatureCodec(32, 20, bits=8, feature_width=16).eval()
    points = torch.rand(2, 32, 3) * 2.0 - 1.0

    reconstruction, features = codec(points, 8)
    permuted_reconstruction, permuted_features = codec(points[:, torch.randperm(32)], 8)
    stream = encode_features(features[0].detach().numpy(), bits=8, output_points=32)
    packet = decode_features(stream)
    decoded = torch.from_numpy(packet.features).to(dtype=points.dtype)[None]

    assert torch.equal(features, permuted_features)
    assert torch.equal(reconstruction, permuted_reconstruction)
    torch.testing.assert_close(codec.decoder(decoded), reconstruction[:1])


def test_feature_codec_benchmark_contract(tmp_path: Path) -> None:
    config = replace(
        FeatureCodecBenchmarkConfig.from_json(
            Path("configs/experiment_018_feature_codec_smoke.json")
        ),
        model_seeds=(3,),
        output_dir=str(tmp_path / "feature"),
    )

    result = run_feature_codec_benchmark(config, device_name="cpu")

    assert result["model_independence"]["all_unique"]
    assert result["per_seed"][0]["model"]["optimizer_updates"] == 2
    assert all(
        row["fresh_chamfer_mse"] >= 0 for row in result["per_seed"][0]["per_cloud"]
    )


def test_stability_experiment_factorial_and_contract(tmp_path: Path) -> None:
    config = replace(
        StabilityExperimentConfig.from_json(
            Path("configs/experiment_019_stability_smoke.json")
        ),
        output_dir=str(tmp_path / "stability"),
    )

    result = run_stability_experiment(config, device_name="cpu")

    assert result["factorial"]["complete"]
    assert result["factorial"]["cells"] == 8
    assert all(result["contract_checks"].values())
    assert result["data_protocol"]["all_partitions_pairwise_disjoint"]
    assert result["stability_gates"]["primary"]["decoder_marginals"] == 2
    assert result["variance_components"][0]["analysis_scale"].startswith("natural_log")
    selection = json.loads(
        (tmp_path / "stability/decoders/seed_1901/selection.json").read_text()
    )
    assert selection["expected_stream_bytes"] == 20
    assert "validation" not in selection and "ood" not in selection


def test_bottleneck_audit_config_and_gate() -> None:
    config = BottleneckAuditConfig(
        num_points=32,
        input_sizes=(16, 32),
        constellation_sizes=(4, 8, 16),
        primary_input_size=32,
        primary_constellation_size=8,
        train_samples=7,
        validation_samples=7,
        parameter_ood_samples=7,
        decoder_epochs=1,
        selector_epochs=1,
        oracle_trials=2,
        free_oracle_steps=2,
    )
    assert config.primary_constellation_size == 8

    runs = []
    for size, rmse in ((4, 0.12), (8, 0.10), (16, 0.09)):
        runs.append(
            {
                "condition": "fps",
                "input_size": 32,
                "constellation_size": size,
                "validation": {"chamfer_rmse": rmse},
                "parameter_ood": {"chamfer_rmse": rmse + 0.01},
            }
        )
    runs.extend(
        [
            {
                "condition": "learned",
                "input_size": 32,
                "constellation_size": 8,
                "validation": {"chamfer_rmse": 0.104},
                "parameter_ood": {"chamfer_rmse": 0.114},
            },
            {
                "condition": "best_subset",
                "input_size": 32,
                "constellation_size": 8,
                "validation": {"chamfer_rmse": 0.10},
                "parameter_ood": {"chamfer_rmse": 0.11},
            },
            {
                "condition": "free_coordinates",
                "input_size": 32,
                "constellation_size": 8,
                "validation": {"chamfer_rmse": 0.09},
                "parameter_ood": {"chamfer_rmse": 0.099},
            },
        ]
    )
    gate = bottleneck_audit_gate(
        runs,
        primary_input_size=32,
        primary_constellation_size=8,
        min_endpoint_improvement_percent=1.0,
        max_adjacent_regression_percent=0.5,
        max_learned_gap_vs_fps_percent=5.0,
        free_coordinate_headroom_percent=5.0,
        decoder_unchanged=True,
    )

    assert gate["passed"]
    assert gate["free_coordinate_headroom_detected"]
    assert gate["validation_learned_gap_vs_fps_percent"] == pytest.approx(4.0)


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


def test_selected_rate_spec_and_gate() -> None:
    config = TrainingConfig(num_points=32, parameter_ood_samples=7)
    spec = SelectedRateSpec(
        config,
        (4, 8, 16),
        ("learned", "relation"),
        min_endpoint_improvement_percent=1.0,
        max_adjacent_regression_percent=0.5,
    )
    assert spec.gated_model_kind == "relation"

    runs = [
        {
            "model_kind": "relation",
            "constellation_size": size,
            "coordinate_payload_bits": 3 * size * 12,
            "validation": {"chamfer_rmse": rmse},
            "parameter_ood": {"chamfer_rmse": rmse + 0.1},
        }
        for size, rmse in ((4, 0.5), (8, 0.49), (16, 0.48))
    ]
    passing = rate_curve_gate(
        runs,
        model_kind="relation",
        split="validation",
        min_endpoint_improvement_percent=1.0,
        max_adjacent_regression_percent=0.5,
    )
    assert passing["passed"]
    assert passing["endpoint_improvement_percent"] == pytest.approx(4.0)

    runs[1]["validation"]["chamfer_rmse"] = 0.51
    failing = rate_curve_gate(
        runs,
        model_kind="relation",
        split="validation",
        min_endpoint_improvement_percent=1.0,
        max_adjacent_regression_percent=0.5,
    )
    assert not failing["passed"]
    assert not failing["adjacency_passed"]


def test_encoder_isolation_spec_and_validation_only_gate() -> None:
    config = TrainingConfig(
        num_points=32,
        constellation_size=8,
        parameter_ood_samples=7,
    )
    spec = EncoderIsolationSpec(config, (0.01, 0.05), (0.1, 1.0))
    assert spec.max_surface_rmse == 0.01

    runs = [
        {
            "condition": "fps",
            "validation": {"chamfer_rmse": 0.1, "surface": 1e-8},
            "parameter_ood": {"chamfer_rmse": 0.11, "surface": 1e-8},
        },
        {
            "condition": "hard_subset",
            "validation": {"chamfer_rmse": 0.104, "surface": 1e-8},
            "parameter_ood": {"chamfer_rmse": 0.114, "surface": 1e-8},
        },
        {
            "condition": "soft_t0p05_s0p1",
            "validation": {"chamfer_rmse": 0.09, "surface": 4e-4},
            "parameter_ood": {"chamfer_rmse": 0.09, "surface": 4e-4},
        },
    ]
    gate = encoder_isolation_gate(
        runs,
        max_surface_rmse=0.01,
        max_rmse_gap_vs_fps_percent=5.0,
    )

    assert gate["passed"]
    assert gate["selected_condition"] == "hard_subset"
    assert gate["validation_rmse_gap_vs_fps_percent"] == pytest.approx(4.0)


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
