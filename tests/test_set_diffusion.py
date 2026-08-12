"""Focused tests for the gated set-diffusion prototype."""

# ruff: noqa: E402

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.data import generate_sample
from pointconstellation.diffusion_experiment import (
    SetDiffusionConfig,
    optimize_target_restarts,
    set_diffusion_experiment,
)
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.set_diffusion import (
    ConditionalSetDenoiser,
    DiffusionSchedule,
    farthest_point_constellation,
    matched_set_rmse,
    multimodality_gate,
    select_best_decoded_candidate,
)
from pointconstellation.quantization import quantization_step


def _points(batch_size: int, num_points: int) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(generate_sample(index, num_points=num_points).points)
            for index in range(batch_size)
        ]
    )


def _denoiser(maximum: int = 8) -> ConditionalSetDenoiser:
    return ConditionalSetDenoiser(
        max_constellation_size=maximum,
        feature_width=16,
        num_heads=4,
        num_layers=1,
    )


def test_denoiser_input_invariance_and_particle_equivariance() -> None:
    torch.manual_seed(3)
    model = _denoiser().eval()
    points = _points(2, 12)
    particles = points[:, :6].clone()
    timesteps = torch.tensor([1, 4])
    output = model(particles, points, timesteps)

    input_permutation = torch.randperm(12)
    particle_permutation = torch.randperm(6)
    input_permuted = model(particles, points[:, input_permutation], timesteps)
    particles_permuted = model(particles[:, particle_permutation], points, timesteps)

    assert torch.allclose(output, input_permuted, atol=2e-6)
    assert torch.allclose(
        output[:, particle_permutation], particles_permuted, atol=2e-6
    )


def test_denoiser_accepts_variable_input_and_constellation_sizes() -> None:
    model = _denoiser().eval()
    for num_points, constellation_size in ((8, 2), (17, 5), (24, 8)):
        points = _points(1, num_points)
        particles = farthest_point_constellation(points, constellation_size)
        output = model(particles, points, torch.tensor([2]))
        assert output.shape == (1, constellation_size, 3)


def test_forward_noising_uses_supplied_noise_and_schedule() -> None:
    schedule = DiffusionSchedule(5, beta_start=0.01, beta_end=0.05)
    clean = torch.full((2, 4, 3), 0.25)
    noise = torch.full_like(clean, -0.5)
    timesteps = torch.tensor([0, 4])
    noisy, returned_noise = schedule.q_sample(clean, timesteps, noise=noise)
    alpha_bar = schedule.alpha_bars[timesteps].reshape(2, 1, 1)
    expected = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

    assert torch.equal(returned_noise, noise)
    assert torch.allclose(noisy, expected)
    assert not torch.equal(noisy, clean)


def test_sampling_starts_from_fps_and_finishes_on_exact_lattice() -> None:
    class ZeroDenoiser(torch.nn.Module):
        def forward(self, particles, points, timesteps):
            return torch.zeros_like(particles)

    points = _points(1, 12)
    schedule = DiffusionSchedule(3)
    sampled = schedule.sample_from_fps(
        ZeroDenoiser(),
        points,
        4,
        bits=8,
        start_step=0,
        stochastic=False,
        initial_noise=torch.zeros(1, 4, 3),
    )
    fps = farthest_point_constellation(points, 4)
    step = quantization_step(8)
    lattice = (sampled + 1.0) / step

    assert torch.allclose(sampled, fps, atol=step)
    assert torch.allclose(lattice, lattice.round(), atol=1e-5)


def test_candidate_selection_scores_through_variable_decoder() -> None:
    torch.manual_seed(11)
    decoder = VariableConstellationDecoder(
        8, 3, feature_width=16, num_heads=4, num_layers=1
    ).eval()
    decoder.requires_grad_(False)
    candidates = torch.tensor(
        [
            [
                [[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]],
                [[0.0, -0.8, 0.0], [0.0, 0.8, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    target = decoder(candidates[:, 1], num_output_points=8)
    best, indices, scores = select_best_decoded_candidate(
        decoder, candidates, target, num_output_points=8
    )

    assert indices.item() == 1
    assert torch.equal(best, candidates[:, 1])
    assert scores[0, 1].item() == pytest.approx(0.0, abs=1e-7)


def test_multimodality_gate_passes_and_fails_on_expected_restarts() -> None:
    base = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    permutation = base.flip(0)
    separated = base + torch.tensor([0.0, 0.4, 0.0])
    assert matched_set_rmse(base, permutation) == pytest.approx(0.0)

    constellations = torch.stack((base, separated))[None]
    passed = multimodality_gate(
        constellations,
        torch.tensor([[0.10, 0.101]]),
        relative_distortion_tolerance=0.02,
        min_matched_separation=0.2,
        min_multimodal_fraction=1.0,
    )
    failed = multimodality_gate(
        constellations,
        torch.tensor([[0.10, 0.20]]),
        relative_distortion_tolerance=0.02,
        min_matched_separation=0.2,
        min_multimodal_fraction=1.0,
    )

    assert passed["passed"]
    assert not failed["passed"]
    assert failed["samples"][0]["comparable_restart_count"] == 1


def test_training_gradient_is_finite_and_target_optimizer_freezes_decoder() -> None:
    torch.manual_seed(13)
    points = _points(2, 8)
    denoiser = _denoiser(4).train()
    schedule = DiffusionSchedule(3)
    clean = farthest_point_constellation(points, 3)
    timesteps = torch.tensor([0, 2])
    noisy, noise = schedule.q_sample(clean, timesteps)
    loss = torch.nn.functional.mse_loss(denoiser(noisy, points, timesteps), noise)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in denoiser.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    decoder = VariableConstellationDecoder(
        8, 4, feature_width=16, num_heads=4, num_layers=1
    ).eval()
    decoder.requires_grad_(False)
    before = {
        name: value.detach().clone() for name, value in decoder.state_dict().items()
    }
    config = SetDiffusionConfig(
        num_points=8,
        input_sizes=(8,),
        constellation_sizes=(3,),
        train_samples=2,
        validation_samples=1,
        batch_size=2,
        decoder_epochs=1,
        target_restarts=2,
        target_steps=1,
        diffusion_epochs=1,
        diffusion_steps=2,
        sampling_start_step=0,
        candidate_count=1,
        feature_width=16,
        num_heads=4,
    )
    restarts, distortions = optimize_target_restarts(decoder, points, points, 3, config)

    assert restarts.shape == (2, 2, 3, 3)
    assert distortions.shape == (2, 2)
    assert torch.isfinite(distortions).all()
    assert all(
        torch.equal(before[name], value) for name, value in decoder.state_dict().items()
    )


def test_tiny_experiment_runs_and_labels_failed_gate(tmp_path) -> None:
    config = SetDiffusionConfig(
        num_points=8,
        input_sizes=(8,),
        constellation_sizes=(2,),
        bits=8,
        train_samples=2,
        validation_samples=1,
        batch_size=1,
        decoder_epochs=1,
        target_restarts=2,
        target_steps=1,
        target_learning_rate=0.02,
        target_initial_noise=0.02,
        diffusion_epochs=1,
        diffusion_learning_rate=1e-3,
        diffusion_steps=2,
        sampling_start_step=0,
        candidate_count=1,
        feature_width=8,
        num_heads=2,
        num_layers=1,
        gate_relative_distortion_tolerance=0.0,
        gate_min_matched_separation=2.0,
        gate_min_multimodal_fraction=1.0,
        output_dir=str(tmp_path),
    )
    result = set_diffusion_experiment(config, device_name="cpu")
    saved = json.loads((tmp_path / "metrics.json").read_text())

    assert result["research_status"] == "experimental_not_justified"
    assert result["warning"]
    assert not result["gate"]["passed"]
    assert result["decoder_frozen_during_target_and_diffusion_training"]
    assert result["decoder_unchanged_after_freeze"]
    assert saved["research_status"] == result["research_status"]
    assert (tmp_path / "decoder.pt").exists()
    assert (tmp_path / "denoiser.pt").exists()
