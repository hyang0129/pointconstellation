"""Run the gated conditional set-diffusion research prototype (Experiment 007)."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pointconstellation.data import ProceduralPointCloudDataset
from pointconstellation.losses import chamfer_squared, pairwise_squared
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.set_diffusion import (
    ConditionalSetDenoiser,
    DiffusionSchedule,
    farthest_point_constellation,
    multimodality_gate,
    select_best_decoded_candidate,
)
from pointconstellation.quantization import quantize_coordinates, quantize_ste
from pointconstellation.train import select_device, set_seed


@dataclass(frozen=True)
class SetDiffusionConfig:
    """Validated configuration for the deliberately small research prototype."""

    num_points: int = 32
    input_sizes: tuple[int, ...] = (24, 32)
    constellation_sizes: tuple[int, ...] = (4, 8)
    bits: int = 10
    train_samples: int = 12
    validation_samples: int = 6
    batch_size: int = 3
    decoder_epochs: int = 2
    target_restarts: int = 3
    target_steps: int = 6
    target_learning_rate: float = 0.04
    target_initial_noise: float = 0.08
    diffusion_epochs: int = 2
    diffusion_learning_rate: float = 1e-3
    diffusion_steps: int = 12
    sampling_start_step: int = 7
    candidate_count: int = 3
    feature_width: int = 48
    num_heads: int = 4
    num_layers: int = 1
    gate_relative_distortion_tolerance: float = 0.03
    gate_min_matched_separation: float = 0.05
    gate_min_multimodal_fraction: float = 0.25
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_007_set_diffusion_smoke"

    def __post_init__(self) -> None:
        if self.num_points < 8:
            raise ValueError("num_points must be at least 8")
        if not self.input_sizes or len(set(self.input_sizes)) != len(self.input_sizes):
            raise ValueError("input_sizes must be nonempty and unique")
        if any(size < 8 or size > self.num_points for size in self.input_sizes):
            raise ValueError("input_sizes must be between 8 and num_points")
        if not self.constellation_sizes or len(set(self.constellation_sizes)) != len(
            self.constellation_sizes
        ):
            raise ValueError("constellation_sizes must be nonempty and unique")
        if any(
            size < 2 or size > min(self.input_sizes)
            for size in self.constellation_sizes
        ):
            raise ValueError("constellation sizes must be between 2 and min input size")
        if self.bits < 2 or self.bits > 24:
            raise ValueError("bits must be between 2 and 24")
        if min(self.train_samples, self.validation_samples, self.batch_size) < 1:
            raise ValueError("sample and batch counts must be positive")
        if (
            min(
                self.decoder_epochs,
                self.target_restarts,
                self.target_steps,
                self.diffusion_epochs,
                self.candidate_count,
            )
            < 1
        ):
            raise ValueError(
                "epoch, restart, step, and candidate counts must be positive"
            )
        if self.target_restarts < 2:
            raise ValueError("target_restarts must be at least 2 for the gate")
        if (
            min(
                self.target_learning_rate,
                self.target_initial_noise,
                self.diffusion_learning_rate,
            )
            <= 0
        ):
            raise ValueError("learning rates and target noise must be positive")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if not 0 <= self.sampling_start_step < self.diffusion_steps:
            raise ValueError("sampling_start_step must lie in the diffusion schedule")
        if self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.gate_relative_distortion_tolerance < 0:
            raise ValueError("gate distortion tolerance cannot be negative")
        if self.gate_min_matched_separation < 0:
            raise ValueError("gate separation cannot be negative")
        if not 0 <= self.gate_min_multimodal_fraction <= 1:
            raise ValueError("gate multimodal fraction must be between zero and one")

    @classmethod
    def from_json(cls, path: Path) -> SetDiffusionConfig:
        values = json.loads(path.read_text())
        for key in ("input_sizes", "constellation_sizes"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def _load_points(config: SetDiffusionConfig, split: str) -> Tensor:
    sample_count = (
        config.train_samples if split == "train" else config.validation_samples
    )
    dataset = ProceduralPointCloudDataset(
        sample_count,
        num_points=config.num_points,
        seed=config.seed,
        split=split,
    )
    loader = DataLoader(dataset, batch_size=sample_count, shuffle=False, num_workers=0)
    return next(iter(loader))["points"]


def _chamfer_per_sample(first: Tensor, second: Tensor) -> Tensor:
    distances = pairwise_squared(first, second)
    return 0.5 * (distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1))


def _operating_points(config: SetDiffusionConfig) -> list[tuple[int, int]]:
    return [
        (input_size, constellation_size)
        for input_size in config.input_sizes
        for constellation_size in config.constellation_sizes
    ]


def _train_decoder(
    decoder: VariableConstellationDecoder,
    points: Tensor,
    config: SetDiffusionConfig,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    operating_points = _operating_points(config)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.decoder_epochs + 1):
        decoder.train()
        order = torch.randperm(len(points), device=points.device)
        loss_sum = 0.0
        count = 0
        for batch_index, start in enumerate(range(0, len(points), config.batch_size)):
            target = points[order[start : start + config.batch_size]]
            input_size, constellation_size = operating_points[
                batch_index % len(operating_points)
            ]
            source = target[:, :input_size]
            constellation = quantize_coordinates(
                farthest_point_constellation(source, constellation_size), config.bits
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction = decoder(constellation, num_output_points=config.num_points)
            loss = chamfer_squared(reconstruction, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)
            count += len(target)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "chamfer_rmse": math.sqrt(loss_sum / count),
        }
        history.append(record)
        print(json.dumps({"stage": "decoder", **record}))
    return history


def optimize_target_restarts(
    decoder: VariableConstellationDecoder,
    source: Tensor,
    target: Tensor,
    constellation_size: int,
    config: SetDiffusionConfig,
) -> tuple[Tensor, Tensor]:
    """Optimize multiple target-like restarts against a frozen decoder."""

    if any(parameter.requires_grad for parameter in decoder.parameters()):
        raise ValueError("decoder must be frozen before target optimization")
    initial = farthest_point_constellation(source, constellation_size)
    restart_constellations = []
    restart_distortions = []
    decoder.eval()
    for _ in range(config.target_restarts):
        coordinates = nn.Parameter(
            (initial + config.target_initial_noise * torch.randn_like(initial)).clamp(
                -1.0, 1.0
            )
        )
        optimizer = torch.optim.Adam((coordinates,), lr=config.target_learning_rate)
        for _ in range(config.target_steps):
            optimizer.zero_grad(set_to_none=True)
            quantized = quantize_ste(
                coordinates, config.bits, training=True, jitter=False
            )
            reconstruction = decoder(quantized, num_output_points=config.num_points)
            losses = _chamfer_per_sample(reconstruction, target)
            losses.mean().backward()
            optimizer.step()
            with torch.no_grad():
                coordinates.clamp_(-1.0, 1.0)
        final = quantize_coordinates(coordinates.detach(), config.bits)
        with torch.no_grad():
            distortion = _chamfer_per_sample(
                decoder(final, num_output_points=config.num_points), target
            )
        restart_constellations.append(final)
        restart_distortions.append(distortion)
    return (
        torch.stack(restart_constellations, dim=1),
        torch.stack(restart_distortions, dim=1),
    )


def _generate_targets(
    decoder: VariableConstellationDecoder,
    points: Tensor,
    config: SetDiffusionConfig,
) -> dict[tuple[int, int], tuple[Tensor, Tensor]]:
    targets = {}
    for input_size, constellation_size in _operating_points(config):
        constellation_batches = []
        distortion_batches = []
        for start in range(0, len(points), config.batch_size):
            target = points[start : start + config.batch_size]
            source = target[:, :input_size]
            constellations, distortions = optimize_target_restarts(
                decoder, source, target, constellation_size, config
            )
            constellation_batches.append(constellations)
            distortion_batches.append(distortions)
        targets[(input_size, constellation_size)] = (
            torch.cat(constellation_batches),
            torch.cat(distortion_batches),
        )
    return targets


def _evaluate_gate(
    targets: dict[tuple[int, int], tuple[Tensor, Tensor]],
    config: SetDiffusionConfig,
) -> dict[str, Any]:
    by_operating_point = {}
    sample_records = []
    for operating_point, (constellations, distortions) in targets.items():
        result = multimodality_gate(
            constellations,
            distortions,
            relative_distortion_tolerance=(config.gate_relative_distortion_tolerance),
            min_matched_separation=config.gate_min_matched_separation,
            min_multimodal_fraction=config.gate_min_multimodal_fraction,
        )
        key = f"N{operating_point[0]}_K{operating_point[1]}"
        by_operating_point[key] = result
        sample_records.extend(result["samples"])
    fraction = sum(record["multimodal"] for record in sample_records) / len(
        sample_records
    )
    passed = fraction >= config.gate_min_multimodal_fraction
    return {
        "passed": passed,
        "reason": (
            None
            if passed
            else "optimized targets lack sufficient evidence of multiple useful modes"
        ),
        "multimodal_sample_fraction": fraction,
        "min_multimodal_fraction": config.gate_min_multimodal_fraction,
        "by_operating_point": by_operating_point,
    }


def _train_denoiser(
    denoiser: ConditionalSetDenoiser,
    schedule: DiffusionSchedule,
    points: Tensor,
    targets: dict[tuple[int, int], tuple[Tensor, Tensor]],
    config: SetDiffusionConfig,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.Adam(
        denoiser.parameters(), lr=config.diffusion_learning_rate
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.diffusion_epochs + 1):
        denoiser.train()
        loss_sum = 0.0
        count = 0
        for input_size, constellation_size in _operating_points(config):
            restarts, distortions = targets[(input_size, constellation_size)]
            best_indices = distortions.argmin(dim=1)
            rows = torch.arange(len(points), device=points.device)
            clean_targets = restarts[rows, best_indices]
            order = torch.randperm(len(points), device=points.device)
            for start in range(0, len(points), config.batch_size):
                indices = order[start : start + config.batch_size]
                source = points[indices, :input_size]
                clean = clean_targets[indices]
                timesteps = torch.randint(
                    0,
                    schedule.num_steps,
                    (len(indices),),
                    device=points.device,
                )
                noisy, noise = schedule.q_sample(clean, timesteps)
                optimizer.zero_grad(set_to_none=True)
                prediction = denoiser(noisy, source, timesteps)
                loss = torch.nn.functional.mse_loss(prediction, noise)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * len(indices)
                count += len(indices)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "noise_mse": loss_sum / count,
        }
        history.append(record)
        print(json.dumps({"stage": "diffusion", **record}))
    return history


@torch.no_grad()
def _evaluate(
    decoder: VariableConstellationDecoder,
    denoiser: ConditionalSetDenoiser,
    schedule: DiffusionSchedule,
    points: Tensor,
    config: SetDiffusionConfig,
) -> dict[str, Any]:
    decoder.eval()
    denoiser.eval()
    results = []
    for input_size, constellation_size in _operating_points(config):
        diffusion_loss_sum = 0.0
        fps_loss_sum = 0.0
        selected_candidate_sum = 0.0
        for start in range(0, len(points), config.batch_size):
            target = points[start : start + config.batch_size]
            source = target[:, :input_size]
            repeated_source = source.repeat_interleave(config.candidate_count, dim=0)
            candidates = schedule.sample_from_fps(
                denoiser,
                repeated_source,
                constellation_size,
                bits=config.bits,
                start_step=config.sampling_start_step,
                stochastic=True,
            ).reshape(len(source), config.candidate_count, constellation_size, 3)
            best, indices, scores = select_best_decoded_candidate(
                decoder,
                candidates,
                target,
                num_output_points=config.num_points,
            )
            diffusion_loss_sum += float(scores.amin(dim=1).sum().item())
            selected_candidate_sum += float(indices.float().sum().item())
            fps = quantize_coordinates(
                farthest_point_constellation(source, constellation_size), config.bits
            )
            fps_reconstruction = decoder(fps, num_output_points=config.num_points)
            fps_loss_sum += float(
                _chamfer_per_sample(fps_reconstruction, target).sum().item()
            )
            expected_step = 2.0 / ((1 << config.bits) - 1)
            lattice = (best + 1.0) / expected_step
            if not torch.allclose(lattice, lattice.round(), atol=2e-5):
                raise RuntimeError("sampler failed exact final coordinate quantization")
        diffusion_rmse = math.sqrt(diffusion_loss_sum / len(points))
        fps_rmse = math.sqrt(fps_loss_sum / len(points))
        results.append(
            {
                "input_size": input_size,
                "constellation_size": constellation_size,
                "diffusion_best_of_candidates_chamfer_rmse": diffusion_rmse,
                "fps_chamfer_rmse": fps_rmse,
                "diffusion_gap_vs_fps_percent": 100.0
                * (diffusion_rmse - fps_rmse)
                / fps_rmse,
                "mean_selected_candidate_index": selected_candidate_sum / len(points),
                "candidate_count": config.candidate_count,
                "coordinate_payload_bits": 3 * constellation_size * config.bits,
            }
        )
    return {"operating_points": results}


def set_diffusion_experiment(
    config: SetDiffusionConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train, gate, and evaluate the prototype; the gate never hides results."""

    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_points = _load_points(config, "train").to(device)
    validation_points = _load_points(config, "validation").to(device)
    decoder = VariableConstellationDecoder(
        config.num_points,
        max(config.constellation_sizes),
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    denoiser = ConditionalSetDenoiser(
        max_constellation_size=max(config.constellation_sizes),
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    schedule = DiffusionSchedule(config.diffusion_steps).to(device)
    started = time.perf_counter()

    decoder_history = _train_decoder(decoder, train_points, config)
    decoder.requires_grad_(False)
    decoder_snapshot = {
        name: value.detach().clone() for name, value in decoder.state_dict().items()
    }
    optimized_targets = _generate_targets(decoder, train_points, config)
    gate = _evaluate_gate(optimized_targets, config)
    status = (
        "multimodality_gate_passed" if gate["passed"] else "experimental_not_justified"
    )
    print(json.dumps({"stage": "multimodality_gate", "status": status, **gate}))

    diffusion_history = _train_denoiser(
        denoiser, schedule, train_points, optimized_targets, config
    )
    evaluation = _evaluate(decoder, denoiser, schedule, validation_points, config)
    decoder_unchanged = all(
        torch.equal(decoder_snapshot[name], value)
        for name, value in decoder.state_dict().items()
    )
    decoder_frozen = not any(
        parameter.requires_grad for parameter in decoder.parameters()
    )
    result = {
        "experiment": "007_set_diffusion",
        "research_status": status,
        "warning": (
            (
                "The configured multimodality gate passed, but target-restart "
                "convergence must be established before treating the modes as "
                "scientific evidence for diffusion."
            )
            if gate["passed"]
            else (
                "The runner executed for engineering validation, but its own "
                "multimodality gate says diffusion is not yet justified."
            )
        ),
        "config": asdict(config),
        "device": str(device),
        "gate": gate,
        "gate_requires_target_convergence_audit": True,
        "decoder_frozen_during_target_and_diffusion_training": decoder_frozen,
        "decoder_unchanged_after_freeze": decoder_unchanged,
        "decoder_history": decoder_history,
        "diffusion_history": diffusion_history,
        "evaluation": evaluation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save({"model": decoder.state_dict()}, output_dir / "decoder.pt")
    torch.save({"model": denoiser.state_dict()}, output_dir / "denoiser.pt")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_007_set_diffusion_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    args = parser.parse_args()
    result = set_diffusion_experiment(
        SetDiffusionConfig.from_json(args.config), device_name=args.device
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
