"""Experiment 006: learn a robust coordinate auto-decoder before amortization."""

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

from pointconstellation.data import generate_sample
from pointconstellation.losses import chamfer_squared, pairwise_squared, repulsion_loss
from pointconstellation.models.coordinate_auto_decoder import (
    COORDINATE_MODES,
    CoordinateAutoDecoder,
    CoordinateOnlyDecoder,
    PermutationInvariantAmortizer,
    project_to_input_surface,
    quantize_constellation,
)
from pointconstellation.train import select_device, set_seed


@dataclass(frozen=True)
class AutoDecoderConfig:
    """Validated configuration for the small, procedural auto-decoder study."""

    num_points: int = 64
    constellation_size: int = 8
    bits: int = 10
    train_samples: int = 14
    heldout_samples: int = 7
    code_restarts: int = 3
    heldout_restarts: int = 4
    alternating_epochs: int = 4
    coordinate_updates_per_epoch: int = 2
    decoder_updates_per_epoch: int = 2
    amortizer_epochs: int = 4
    heldout_steps: int = 12
    coordinate_learning_rate: float = 0.04
    decoder_learning_rate: float = 1e-3
    amortizer_learning_rate: float = 2e-3
    heldout_learning_rate: float = 0.04
    imitation_weight: float = 0.25
    repulsion_weight: float = 0.001
    noise_scales: tuple[float, ...] = (0.0, 0.01, 0.03)
    modes: tuple[str, ...] = COORDINATE_MODES
    feature_width: int = 48
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_006_auto_decoder_smoke"

    def __post_init__(self) -> None:
        if self.num_points < 8:
            raise ValueError("num_points must be at least 8")
        if not 2 <= self.constellation_size <= self.num_points:
            raise ValueError("constellation_size must be between 2 and num_points")
        if not 2 <= self.bits <= 24:
            raise ValueError("bits must be between 2 and 24")
        if self.train_samples < 1 or self.heldout_samples < 1:
            raise ValueError("train_samples and heldout_samples must be positive")
        if self.code_restarts < 2 or self.heldout_restarts < 2:
            raise ValueError("both restart counts must be at least two")
        integer_counts = (
            self.alternating_epochs,
            self.coordinate_updates_per_epoch,
            self.decoder_updates_per_epoch,
            self.amortizer_epochs,
            self.heldout_steps,
        )
        if any(value < 1 for value in integer_counts):
            raise ValueError("training epoch and update counts must be positive")
        rates = (
            self.coordinate_learning_rate,
            self.decoder_learning_rate,
            self.amortizer_learning_rate,
            self.heldout_learning_rate,
        )
        if any(value <= 0 for value in rates):
            raise ValueError("learning rates must be positive")
        if self.imitation_weight < 0 or self.repulsion_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if not self.noise_scales or any(scale < 0 for scale in self.noise_scales):
            raise ValueError("noise_scales must be nonempty and nonnegative")
        if 0.0 not in self.noise_scales:
            raise ValueError("noise_scales must include the clean scale 0.0")
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must be nonempty and unique")
        if any(mode not in COORDINATE_MODES for mode in self.modes):
            raise ValueError(f"modes must be drawn from {COORDINATE_MODES}")
        if self.feature_width < 8:
            raise ValueError("feature_width must be at least 8")

    @classmethod
    def from_json(cls, path: Path) -> AutoDecoderConfig:
        values = json.loads(path.read_text())
        for key in ("noise_scales", "modes"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class HeldoutInference:
    """Tensor-valued result used by tests and the JSON metric adapter."""

    one_shot_coordinates: Tensor
    refined_coordinates: Tensor
    all_restart_coordinates: Tensor
    initial_restart_coordinates: Tensor
    one_shot_losses: Tensor
    refined_losses: Tensor
    decoder_unchanged: bool


def _procedural_tensor(
    count: int,
    *,
    num_points: int,
    seed: int,
    split: str,
    device: torch.device,
) -> Tensor:
    return torch.stack(
        [
            torch.from_numpy(
                generate_sample(
                    sample_id,
                    num_points=num_points,
                    seed=seed,
                    split=split,
                ).points
            )
            for sample_id in range(count)
        ]
    ).to(device)


def _chamfer_per_sample(first: Tensor, second: Tensor) -> Tensor:
    distances = pairwise_squared(first, second)
    return 0.5 * (distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1))


def _decode_restarts(
    decoder: CoordinateOnlyDecoder,
    coordinates: Tensor,
) -> Tensor:
    batch_size, restarts, _, _ = coordinates.shape
    return decoder(coordinates.flatten(0, 1)).reshape(
        batch_size,
        restarts,
        decoder.num_output_points,
        3,
    )


def _restart_losses(reconstruction: Tensor, target: Tensor) -> Tensor:
    batch_size, restarts, num_points, _ = reconstruction.shape
    expanded_target = target[:, None].expand(-1, restarts, -1, -1)
    return _chamfer_per_sample(
        reconstruction.reshape(batch_size * restarts, num_points, 3),
        expanded_target.reshape(batch_size * restarts, target.shape[1], 3),
    ).reshape(batch_size, restarts)


def _best_restart(values: Tensor, losses: Tensor) -> Tensor:
    indices = losses.argmin(dim=1)
    batch_indices = torch.arange(len(values), device=values.device)
    return values[batch_indices, indices]


def _set_metrics(
    reconstruction: Tensor,
    target: Tensor,
    constellation: Tensor,
) -> dict[str, float]:
    reconstruction_distances = pairwise_squared(reconstruction, target)
    chamfer = 0.5 * (
        reconstruction_distances.amin(dim=2).mean(dim=1)
        + reconstruction_distances.amin(dim=1).mean(dim=1)
    )
    anchor_distances = pairwise_squared(constellation, target)
    surface = anchor_distances.amin(dim=2).mean(dim=1)
    coverage = anchor_distances.amin(dim=1).mean(dim=1)
    return {
        "count": len(target),
        "chamfer": float(chamfer.mean().item()),
        "chamfer_rmse": math.sqrt(float(chamfer.mean().item())),
        "surface": float(surface.mean().item()),
        "surface_rmse": math.sqrt(float(surface.mean().item())),
        "coverage": float(coverage.mean().item()),
        "coverage_rmse": math.sqrt(float(coverage.mean().item())),
    }


def _evaluated_coordinates(
    raw: Tensor,
    points: Tensor,
    config: AutoDecoderConfig,
    mode: str,
    *,
    straight_through: bool,
) -> Tensor:
    coordinates = raw + (raw.clamp(-1.0, 1.0) - raw).detach()
    if mode == "projected":
        coordinates = project_to_input_surface(
            coordinates,
            points,
            straight_through=straight_through,
        )
    return quantize_constellation(
        coordinates,
        config.bits,
        straight_through=straight_through,
        jitter=False,
    )


def _initial_heldout_restarts(
    amortized: Tensor,
    points: Tensor,
    config: AutoDecoderConfig,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 6001)
    starts = [amortized]
    for _ in range(1, config.heldout_restarts):
        per_cloud = []
        for cloud in points:
            indices = torch.randperm(len(cloud), generator=generator)[
                : config.constellation_size
            ].to(points.device)
            per_cloud.append(cloud[indices])
        starts.append(torch.stack(per_cloud))
    return torch.stack(starts, dim=1)


def infer_heldout_coordinates(
    decoder: CoordinateOnlyDecoder,
    amortizer: PermutationInvariantAmortizer,
    points: Tensor,
    config: AutoDecoderConfig,
    *,
    mode: str,
) -> HeldoutInference:
    """Infer held-out literal coordinates while keeping the decoder bitwise fixed."""

    if mode not in COORDINATE_MODES:
        raise ValueError(f"mode must be one of {COORDINATE_MODES}")
    decoder_before = {
        name: value.detach().clone() for name, value in decoder.state_dict().items()
    }
    decoder_training = decoder.training
    decoder_requires_grad = [
        parameter.requires_grad for parameter in decoder.parameters()
    ]
    decoder.eval().requires_grad_(False)
    amortizer.eval()

    with torch.no_grad():
        amortized = amortizer(points)
    raw = nn.Parameter(_initial_heldout_restarts(amortized, points, config))
    optimizer = torch.optim.Adam((raw,), lr=config.heldout_learning_rate)

    with torch.no_grad():
        initial = _evaluated_coordinates(
            raw,
            points,
            config,
            mode,
            straight_through=False,
        )
        initial_losses = _restart_losses(_decode_restarts(decoder, initial), points)
        best_coordinates = initial.detach().clone()
        best_losses = initial_losses.detach().clone()

    for _ in range(config.heldout_steps):
        optimizer.zero_grad(set_to_none=True)
        coordinates = _evaluated_coordinates(
            raw,
            points,
            config,
            mode,
            straight_through=True,
        )
        reconstruction = _decode_restarts(decoder, coordinates)
        loss = _restart_losses(reconstruction, points).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            raw.clamp_(-1.0, 1.0)
            evaluated = _evaluated_coordinates(
                raw,
                points,
                config,
                mode,
                straight_through=False,
            )
            evaluated_losses = _restart_losses(
                _decode_restarts(decoder, evaluated), points
            )
            improved = evaluated_losses < best_losses
            best_losses = torch.where(improved, evaluated_losses, best_losses)
            best_coordinates = torch.where(
                improved[:, :, None, None], evaluated, best_coordinates
            )

    one_shot = initial[:, 0]
    one_shot_losses = initial_losses[:, 0]
    refined = _best_restart(best_coordinates, best_losses)
    refined_losses = best_losses.amin(dim=1)
    unchanged = all(
        torch.equal(value, decoder_before[name])
        for name, value in decoder.state_dict().items()
    )
    for parameter, requires_grad in zip(
        decoder.parameters(), decoder_requires_grad, strict=True
    ):
        parameter.requires_grad_(requires_grad)
    decoder.train(decoder_training)
    return HeldoutInference(
        one_shot_coordinates=one_shot.detach(),
        refined_coordinates=refined.detach(),
        all_restart_coordinates=best_coordinates.detach(),
        initial_restart_coordinates=initial.detach(),
        one_shot_losses=one_shot_losses.detach(),
        refined_losses=refined_losses.detach(),
        decoder_unchanged=unchanged,
    )


def _parameter_delta(before: dict[str, Tensor], module: nn.Module) -> float:
    total = 0.0
    for name, value in module.state_dict().items():
        total += float((value.detach() - before[name]).square().sum().item())
    return math.sqrt(total)


def _restart_diversity(coordinates: Tensor) -> float:
    if coordinates.shape[1] < 2:
        return 0.0
    pair_distances = []
    for first in range(coordinates.shape[1]):
        for second in range(first + 1, coordinates.shape[1]):
            pair_distances.append(
                chamfer_squared(coordinates[:, first], coordinates[:, second])
            )
    return math.sqrt(float(torch.stack(pair_distances).mean().item()))


def _train_alternating(
    model: CoordinateAutoDecoder,
    train_points: Tensor,
    config: AutoDecoderConfig,
    mode: str,
) -> list[dict[str, float]]:
    indices = torch.arange(len(train_points), device=train_points.device)
    coordinate_optimizer = torch.optim.Adam(
        model.codes.parameters(), lr=config.coordinate_learning_rate
    )
    decoder_optimizer = torch.optim.Adam(
        model.decoder.parameters(), lr=config.decoder_learning_rate
    )
    history = []

    for epoch in range(1, config.alternating_epochs + 1):
        model.decoder.eval().requires_grad_(False)
        model.codes.train().requires_grad_(True)
        coordinate_loss = torch.tensor(float("nan"), device=train_points.device)
        for _ in range(config.coordinate_updates_per_epoch):
            coordinate_optimizer.zero_grad(set_to_none=True)
            coordinates = model.codes(
                indices,
                reference_points=train_points if mode == "projected" else None,
                straight_through=True,
                jitter=False,
            )
            reconstruction = _decode_restarts(model.decoder, coordinates)
            coordinate_loss = _restart_losses(reconstruction, train_points).mean()
            repulsion = repulsion_loss(coordinates.flatten(0, 1))
            (coordinate_loss + config.repulsion_weight * repulsion).backward()
            coordinate_optimizer.step()
            with torch.no_grad():
                model.codes.coordinates.clamp_(-1.0, 1.0)

        model.codes.eval().requires_grad_(False)
        model.decoder.train().requires_grad_(True)
        decoder_loss = torch.tensor(float("nan"), device=train_points.device)
        for _ in range(config.decoder_updates_per_epoch):
            with torch.no_grad():
                clean = model.codes(
                    indices,
                    reference_points=train_points if mode == "projected" else None,
                    straight_through=False,
                )
            decoder_optimizer.zero_grad(set_to_none=True)
            neighborhood_losses = []
            for scale in config.noise_scales:
                if scale:
                    noisy = (clean + torch.randn_like(clean) * scale).clamp(-1.0, 1.0)
                else:
                    noisy = clean
                neighborhood_losses.append(
                    _restart_losses(
                        _decode_restarts(model.decoder, noisy), train_points
                    ).mean()
                )
            decoder_loss = torch.stack(neighborhood_losses).mean()
            decoder_loss.backward()
            decoder_optimizer.step()

        with torch.no_grad():
            exact = model.codes(
                indices,
                reference_points=train_points if mode == "projected" else None,
                straight_through=False,
            )
            exact_loss = (
                _restart_losses(_decode_restarts(model.decoder, exact), train_points)
                .amin(dim=1)
                .mean()
            )
        record = {
            "epoch": epoch,
            "coordinate_chamfer_rmse": math.sqrt(float(coordinate_loss.item())),
            "decoder_neighborhood_chamfer_rmse": math.sqrt(float(decoder_loss.item())),
            "best_restart_clean_chamfer_rmse": math.sqrt(float(exact_loss.item())),
        }
        history.append(record)
        print(json.dumps({"stage": "alternating", "mode": mode, **record}))

    model.codes.requires_grad_(True)
    model.decoder.requires_grad_(True)
    return history


def _train_amortizer(
    amortizer: PermutationInvariantAmortizer,
    decoder: CoordinateOnlyDecoder,
    model: CoordinateAutoDecoder,
    train_points: Tensor,
    config: AutoDecoderConfig,
    mode: str,
) -> list[dict[str, float]]:
    indices = torch.arange(len(train_points), device=train_points.device)
    decoder.eval().requires_grad_(False)
    with torch.no_grad():
        codes = model.codes(
            indices,
            reference_points=train_points if mode == "projected" else None,
            straight_through=False,
        )
        losses = _restart_losses(_decode_restarts(decoder, codes), train_points)
        targets = _best_restart(codes, losses)
    optimizer = torch.optim.Adam(
        amortizer.parameters(), lr=config.amortizer_learning_rate
    )
    history = []
    for epoch in range(1, config.amortizer_epochs + 1):
        amortizer.train()
        optimizer.zero_grad(set_to_none=True)
        raw = amortizer(train_points)
        predicted = _evaluated_coordinates(
            raw,
            train_points,
            config,
            mode,
            straight_through=True,
        )
        reconstruction = decoder(predicted)
        reconstruction_loss = chamfer_squared(reconstruction, train_points)
        imitation_loss = chamfer_squared(predicted, targets)
        loss = reconstruction_loss + config.imitation_weight * imitation_loss
        loss.backward()
        optimizer.step()
        record = {
            "epoch": epoch,
            "reconstruction_chamfer_rmse": math.sqrt(float(reconstruction_loss.item())),
            "code_imitation_chamfer_rmse": math.sqrt(float(imitation_loss.item())),
        }
        history.append(record)
        print(json.dumps({"stage": "amortizer", "mode": mode, **record}))
    decoder.requires_grad_(True)
    return history


def _evaluate_trained_codes(
    model: CoordinateAutoDecoder,
    train_points: Tensor,
    config: AutoDecoderConfig,
    mode: str,
) -> tuple[dict[str, float], dict[str, Any], Tensor]:
    model.eval()
    indices = torch.arange(len(train_points), device=train_points.device)
    with torch.no_grad():
        all_codes = model.codes(
            indices,
            reference_points=train_points if mode == "projected" else None,
            straight_through=False,
        )
        clean_reconstruction = _decode_restarts(model.decoder, all_codes)
        losses = _restart_losses(clean_reconstruction, train_points)
        codes = _best_restart(all_codes, losses)
        clean = _set_metrics(model.decoder(codes), train_points, codes)
        by_scale = {}
        for scale in config.noise_scales:
            generator = torch.Generator(device="cpu").manual_seed(
                config.seed + 7001 + round(scale * 1_000_000)
            )
            if scale:
                noise = torch.randn(codes.shape, generator=generator).to(codes.device)
                evaluated = (codes + scale * noise).clamp(-1.0, 1.0)
            else:
                evaluated = codes
            by_scale[str(scale)] = _set_metrics(
                model.decoder(evaluated), train_points, evaluated
            )
    nonzero = [values["chamfer"] for key, values in by_scale.items() if key != "0.0"]
    noisy = {
        "by_scale": by_scale,
        "mean_nonzero_chamfer_rmse": (
            math.sqrt(sum(nonzero) / len(nonzero)) if nonzero else clean["chamfer_rmse"]
        ),
    }
    return clean, noisy, all_codes


def _run_mode(
    config: AutoDecoderConfig,
    mode: str,
    train_points: Tensor,
    heldout_points: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(config.seed)
    model = CoordinateAutoDecoder(
        num_clouds=config.train_samples,
        num_restarts=config.code_restarts,
        num_output_points=config.num_points,
        constellation_size=config.constellation_size,
        bits=config.bits,
        mode=mode,
        feature_width=config.feature_width,
        initial_clouds=train_points,
        seed=config.seed,
    ).to(device)
    decoder_initial = {
        name: value.detach().clone()
        for name, value in model.decoder.state_dict().items()
    }
    code_initial = model.codes.coordinates.detach().clone()
    alternating_history = _train_alternating(model, train_points, config, mode)

    amortizer = PermutationInvariantAmortizer(
        config.constellation_size,
        feature_width=config.feature_width,
    ).to(device)
    amortizer_history = _train_amortizer(
        amortizer,
        model.decoder,
        model,
        train_points,
        config,
        mode,
    )
    clean, noisy, all_train_codes = _evaluate_trained_codes(
        model, train_points, config, mode
    )
    inference = infer_heldout_coordinates(
        model.decoder,
        amortizer,
        heldout_points,
        config,
        mode=mode,
    )
    with torch.no_grad():
        one_shot_reconstruction = model.decoder(inference.one_shot_coordinates)
        refined_reconstruction = model.decoder(inference.refined_coordinates)
        one_shot = _set_metrics(
            one_shot_reconstruction,
            heldout_points,
            inference.one_shot_coordinates,
        )
        refined = _set_metrics(
            refined_reconstruction,
            heldout_points,
            inference.refined_coordinates,
        )
    improvement = (
        100.0
        * (one_shot["chamfer_rmse"] - refined["chamfer_rmse"])
        / one_shot["chamfer_rmse"]
    )
    initial_restart_diversity = _restart_diversity(
        inference.initial_restart_coordinates
    )
    behavior = {
        "representation": "literal_quantized_Kx3_coordinates_only",
        "coordinate_mode": mode,
        "literal_code_shape": list(model.codes.coordinates.shape),
        "coordinate_parameter_count": model.codes.coordinates.numel(),
        "decoder_parameter_count": sum(
            parameter.numel() for parameter in model.decoder.parameters()
        ),
        "amortizer_parameter_count": sum(
            parameter.numel() for parameter in amortizer.parameters()
        ),
        "coordinate_parameter_delta_l2": float(
            (model.codes.coordinates.detach() - code_initial)
            .square()
            .sum()
            .sqrt()
            .item()
        ),
        "decoder_parameter_delta_l2": _parameter_delta(decoder_initial, model.decoder),
        "train_restart_diversity_chamfer_rmse": _restart_diversity(all_train_codes),
        "heldout_initial_restart_diversity_chamfer_rmse": initial_restart_diversity,
        "heldout_decoder_unchanged": inference.decoder_unchanged,
        "alternating_coordinate_decoder_updates": True,
        "straight_through_quantization_during_optimization": True,
        "exact_quantization_during_evaluation": True,
        "noise_neighborhood_scales": list(config.noise_scales),
    }
    return {
        "mode": mode,
        "history": {
            "alternating": alternating_history,
            "amortizer": amortizer_history,
        },
        "clean": clean,
        "noisy": noisy,
        "heldout": {
            "one_shot": one_shot,
            "refined": refined,
            "refinement_improvement_percent": improvement,
            "one_shot_mean_chamfer": float(inference.one_shot_losses.mean().item()),
            "refined_mean_chamfer": float(inference.refined_losses.mean().item()),
            "decoder_unchanged": inference.decoder_unchanged,
        },
        "behavior": behavior,
    }


def run_auto_decoder_experiment(
    config: AutoDecoderConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run unrestricted/projected auto-decoder studies and write metrics JSON."""

    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_points = _procedural_tensor(
        config.train_samples,
        num_points=config.num_points,
        seed=config.seed,
        split="train",
        device=device,
    )
    heldout_points = _procedural_tensor(
        config.heldout_samples,
        num_points=config.num_points,
        seed=config.seed,
        split="validation",
        device=device,
    )
    started = time.perf_counter()
    runs = {
        mode: _run_mode(config, mode, train_points, heldout_points, device)
        for mode in config.modes
    }
    result = {
        "experiment": "006_coordinate_auto_decoder",
        "config": asdict(config),
        "device": str(device),
        "runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    result["metrics_path"] = str(metrics_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_006_auto_decoder_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    args = parser.parse_args()
    result = run_auto_decoder_experiment(
        AutoDecoderConfig.from_json(args.config),
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
