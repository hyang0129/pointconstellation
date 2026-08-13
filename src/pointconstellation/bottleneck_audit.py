"""Run Experiment 004's frozen-decoder coordinate bottleneck audit."""

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
from pointconstellation.models import (
    ProgressiveSubsetEncoder,
    VariableConstellationDecoder,
)
from pointconstellation.quantization import quantize_coordinates, quantize_ste
from pointconstellation.train import select_device, set_seed

SAMPLERS = ("fps", "random", "normal_fps")
CONDITIONS = ("fps", "random", "learned")


@dataclass(frozen=True)
class BottleneckAuditConfig:
    num_points: int = 256
    input_sizes: tuple[int, ...] = (64, 128, 256)
    constellation_sizes: tuple[int, ...] = (4, 8, 16, 32)
    bits: int = 12
    train_samples: int = 448
    validation_samples: int = 140
    parameter_ood_samples: int = 140
    batch_size: int = 8
    decoder_epochs: int = 12
    selector_epochs: int = 12
    decoder_learning_rate: float = 1e-3
    selector_learning_rate: float = 1e-3
    feature_width: int = 96
    num_heads: int = 4
    num_layers: int = 2
    selection_temperature: float = 0.1
    training_samplers: tuple[str, ...] = SAMPLERS
    primary_input_size: int = 256
    primary_constellation_size: int = 16
    oracle_trials: int = 8
    free_oracle_steps: int = 16
    free_oracle_learning_rate: float = 0.03
    min_endpoint_improvement_percent: float = 1.0
    max_adjacent_regression_percent: float = 0.5
    max_learned_gap_vs_fps_percent: float = 5.0
    free_coordinate_headroom_percent: float = 5.0
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_004_frozen_decoder"

    def __post_init__(self) -> None:
        if self.num_points < 8:
            raise ValueError("num_points must be at least 8")
        if not self.input_sizes or len(set(self.input_sizes)) != len(self.input_sizes):
            raise ValueError("input_sizes must be nonempty and unique")
        if not self.constellation_sizes or len(set(self.constellation_sizes)) != len(
            self.constellation_sizes
        ):
            raise ValueError("constellation_sizes must be nonempty and unique")
        if any(size < 8 or size > self.num_points for size in self.input_sizes):
            raise ValueError("input sizes must be between 8 and num_points")
        if any(
            size < 2 or size > min(self.input_sizes)
            for size in self.constellation_sizes
        ):
            raise ValueError(
                "constellation sizes must be between 2 and the smallest input size"
            )
        if self.primary_input_size not in self.input_sizes:
            raise ValueError("primary_input_size must be one of input_sizes")
        if self.primary_constellation_size not in self.constellation_sizes:
            raise ValueError(
                "primary_constellation_size must be one of constellation_sizes"
            )
        if not self.training_samplers or any(
            sampler not in SAMPLERS for sampler in self.training_samplers
        ):
            raise ValueError(f"training_samplers must be drawn from {SAMPLERS}")
        if self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if self.decoder_epochs < 1 or self.selector_epochs < 1:
            raise ValueError("decoder and selector epochs must be positive")
        if self.parameter_ood_samples < 1:
            raise ValueError("parameter_ood_samples must be positive")
        if self.oracle_trials < 1 or self.free_oracle_steps < 1:
            raise ValueError("oracle trials and steps must be positive")

    @classmethod
    def from_json(cls, path: Path) -> BottleneckAuditConfig:
        values = json.loads(path.read_text())
        for key in ("input_sizes", "constellation_sizes", "training_samplers"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def _make_loaders(
    config: BottleneckAuditConfig,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
]:
    datasets = {
        "train": ProceduralPointCloudDataset(
            config.train_samples,
            num_points=config.num_points,
            seed=config.seed,
            split="train",
        ),
        "validation": ProceduralPointCloudDataset(
            config.validation_samples,
            num_points=config.num_points,
            seed=config.seed,
            split="validation",
        ),
        "parameter_ood": ProceduralPointCloudDataset(
            config.parameter_ood_samples,
            num_points=config.num_points,
            seed=config.seed,
            split="parameter_ood",
        ),
    }
    generator = torch.Generator().manual_seed(config.seed)
    return (
        DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        ),
        DataLoader(
            datasets["validation"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        DataLoader(
            datasets["parameter_ood"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    )


def _gather_points(points: Tensor, indices: Tensor) -> Tensor:
    return points.gather(1, indices[:, :, None].expand(-1, -1, 3))


def _fps_subset(points: Tensor, constellation_size: int, bits: int) -> Tensor:
    batch_size, num_points, _ = points.shape
    batch_indices = torch.arange(batch_size, device=points.device)
    centroid = points.mean(dim=1, keepdim=True)
    farthest = ((points - centroid) ** 2).sum(dim=-1).argmax(dim=1)
    minimum_distances = torch.full(
        (batch_size, num_points),
        torch.inf,
        dtype=points.dtype,
        device=points.device,
    )
    selected_indices = torch.empty(
        (batch_size, constellation_size),
        dtype=torch.long,
        device=points.device,
    )
    for anchor_index in range(constellation_size):
        selected_indices[:, anchor_index] = farthest
        selected = points[batch_indices, farthest][:, None, :]
        minimum_distances = torch.minimum(
            minimum_distances, ((points - selected) ** 2).sum(dim=-1)
        )
        farthest = minimum_distances.argmax(dim=1)
    return quantize_coordinates(_gather_points(points, selected_indices), bits)


def _random_subset(points: Tensor, constellation_size: int, bits: int) -> Tensor:
    priorities = torch.rand(points.shape[:2], device=points.device)
    indices = priorities.topk(constellation_size, dim=1).indices
    return quantize_coordinates(_gather_points(points, indices), bits)


def _normal_fps_subset(
    points: Tensor, normals: Tensor, constellation_size: int, bits: int
) -> Tensor:
    batch_size, num_points, _ = points.shape
    neighbor_count = min(8, num_points - 1)
    distances = pairwise_squared(points, points)
    diagonal = torch.eye(num_points, dtype=torch.bool, device=points.device)[None]
    neighbor_indices = (
        distances.masked_fill(diagonal, torch.inf)
        .topk(neighbor_count, dim=2, largest=False)
        .indices
    )
    expanded_normals = normals[:, None, :, :].expand(-1, num_points, -1, -1)
    neighbors = expanded_normals.gather(
        2, neighbor_indices[:, :, :, None].expand(-1, -1, -1, 3)
    )
    alignment = (normals[:, :, None, :] * neighbors).sum(dim=-1).abs()
    variation = 1.0 - alignment.mean(dim=2)

    batch_indices = torch.arange(batch_size, device=points.device)
    farthest = variation.argmax(dim=1)
    minimum_distances = torch.full(
        (batch_size, num_points),
        torch.inf,
        dtype=points.dtype,
        device=points.device,
    )
    selected_indices = torch.empty(
        (batch_size, constellation_size),
        dtype=torch.long,
        device=points.device,
    )
    for anchor_index in range(constellation_size):
        selected_indices[:, anchor_index] = farthest
        selected = points[batch_indices, farthest][:, None, :]
        minimum_distances = torch.minimum(
            minimum_distances, ((points - selected) ** 2).sum(dim=-1)
        )
        priority = minimum_distances * (1.0 + variation)
        farthest = priority.argmax(dim=1)
    return quantize_coordinates(_gather_points(points, selected_indices), bits)


def _sample_subset(
    points: Tensor,
    normals: Tensor,
    constellation_size: int,
    bits: int,
    sampler: str,
) -> Tensor:
    if sampler == "fps":
        return _fps_subset(points, constellation_size, bits)
    if sampler == "random":
        return _random_subset(points, constellation_size, bits)
    if sampler == "normal_fps":
        return _normal_fps_subset(points, normals, constellation_size, bits)
    raise ValueError(f"unknown sampler: {sampler}")


def _training_source(
    points: Tensor, normals: Tensor, input_size: int
) -> tuple[Tensor, Tensor]:
    permutation = torch.randperm(points.shape[1], device=points.device)[:input_size]
    return points[:, permutation], normals[:, permutation]


def _chamfer_per_sample(first: Tensor, second: Tensor) -> Tensor:
    distances = pairwise_squared(first, second)
    return 0.5 * (distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1))


def _per_sample_metrics(
    reconstruction: Tensor, target: Tensor, constellation: Tensor
) -> dict[str, Tensor]:
    reconstruction_target = pairwise_squared(reconstruction, target)
    forward = reconstruction_target.amin(dim=2)
    backward = reconstruction_target.amin(dim=1)
    chamfer = 0.5 * (forward.mean(dim=1) + backward.mean(dim=1))
    hausdorff = torch.maximum(forward.amax(dim=1), backward.amax(dim=1)).sqrt()

    anchor_target = pairwise_squared(constellation, target)
    surface = anchor_target.amin(dim=2).mean(dim=1)
    coverage = anchor_target.amin(dim=1).mean(dim=1)
    preservation = (
        pairwise_squared(constellation, reconstruction).amin(dim=2).mean(dim=1)
    )
    anchor_distances = pairwise_squared(constellation, constellation)
    diagonal = torch.eye(
        constellation.shape[1], dtype=torch.bool, device=constellation.device
    )[None]
    separation = anchor_distances.masked_fill(diagonal, torch.inf).amin(dim=2)
    return {
        "chamfer": chamfer,
        "hausdorff": hausdorff,
        "coverage": coverage,
        "surface": surface,
        "preservation": preservation,
        "minimum_anchor_separation": separation.sqrt().mean(dim=1),
    }


def _empty_accumulator() -> dict[str, float]:
    return {
        "count": 0.0,
        "chamfer": 0.0,
        "hausdorff": 0.0,
        "coverage": 0.0,
        "surface": 0.0,
        "preservation": 0.0,
        "minimum_anchor_separation": 0.0,
    }


def _accumulate_samples(
    totals: dict[str, float], metrics: dict[str, Tensor], indices: Tensor | None = None
) -> None:
    selected = (
        metrics
        if indices is None
        else {name: values[indices] for name, values in metrics.items()}
    )
    count = len(next(iter(selected.values())))
    totals["count"] += count
    for name, values in selected.items():
        totals[name] += float(values.sum().item())


def _finish_metrics(totals: dict[str, float]) -> dict[str, float]:
    count = totals["count"]
    result = {name: value / count for name, value in totals.items() if name != "count"}
    for name in ("chamfer", "coverage", "surface", "preservation"):
        result[f"{name}_rmse"] = math.sqrt(result[name])
    return result


def _best_sampled_subset(
    decoder: VariableConstellationDecoder,
    source: Tensor,
    target: Tensor,
    constellation_size: int,
    config: BottleneckAuditConfig,
) -> Tensor:
    candidates = [_fps_subset(source, constellation_size, config.bits)]
    candidates.extend(
        _random_subset(source, constellation_size, config.bits)
        for _ in range(config.oracle_trials)
    )
    losses = torch.stack(
        [
            _chamfer_per_sample(
                decoder(candidate, num_output_points=config.num_points), target
            )
            for candidate in candidates
        ],
        dim=1,
    )
    best = losses.argmin(dim=1)
    stacked = torch.stack(candidates, dim=1)
    batch_indices = torch.arange(len(source), device=source.device)
    return stacked[batch_indices, best]


def _free_coordinate_oracle(
    decoder: VariableConstellationDecoder,
    initial: Tensor,
    target: Tensor,
    config: BottleneckAuditConfig,
) -> Tensor:
    free_coordinates = nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.Adam(
        (free_coordinates,), lr=config.free_oracle_learning_rate
    )
    with torch.enable_grad():
        for _ in range(config.free_oracle_steps):
            optimizer.zero_grad(set_to_none=True)
            quantized = quantize_ste(
                free_coordinates, config.bits, training=True, jitter=False
            )
            reconstruction = decoder(quantized, num_output_points=config.num_points)
            loss = _chamfer_per_sample(reconstruction, target).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                free_coordinates.clamp_(-1.0, 1.0)
    return quantize_coordinates(free_coordinates.detach(), config.bits)


def _evaluate_condition(
    decoder: VariableConstellationDecoder,
    selector: ProgressiveSubsetEncoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: BottleneckAuditConfig,
    *,
    condition: str,
    input_size: int,
    constellation_size: int,
    seed_offset: int,
) -> dict[str, Any]:
    decoder.eval()
    selector.eval()
    set_seed(config.seed + seed_offset)
    totals = _empty_accumulator()
    family_totals: dict[str, dict[str, float]] = {}
    for batch in loader:
        target = batch["points"].to(device)
        source = target[:, :input_size]
        with torch.no_grad():
            if condition == "fps":
                constellation = _fps_subset(source, constellation_size, config.bits)
            elif condition == "random":
                constellation = _random_subset(source, constellation_size, config.bits)
            elif condition == "learned":
                constellation = selector(source, constellation_size)
            elif condition in ("best_subset", "free_coordinates"):
                constellation = _best_sampled_subset(
                    decoder, source, target, constellation_size, config
                )
            else:
                raise ValueError(f"unknown condition: {condition}")

        if condition == "free_coordinates":
            constellation = _free_coordinate_oracle(
                decoder, constellation, target, config
            )

        with torch.no_grad():
            reconstruction = decoder(constellation, num_output_points=config.num_points)
            metrics = _per_sample_metrics(reconstruction, target, constellation)
        _accumulate_samples(totals, metrics)
        families = batch["family"]
        for family in sorted(set(families)):
            indices = torch.tensor(
                [index for index, value in enumerate(families) if value == family],
                dtype=torch.long,
                device=device,
            )
            family_total = family_totals.setdefault(family, _empty_accumulator())
            _accumulate_samples(family_total, metrics, indices)

    result: dict[str, Any] = _finish_metrics(totals)
    result["by_family"] = {
        family: _finish_metrics(values)
        for family, values in sorted(family_totals.items())
    }
    return result


def _quick_chamfer(
    decoder: VariableConstellationDecoder,
    selector: ProgressiveSubsetEncoder | None,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: BottleneckAuditConfig,
    *,
    condition: str,
) -> float:
    decoder.eval()
    if selector is not None:
        selector.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["points"].to(device)
            source = target[:, : config.primary_input_size]
            if condition == "fps":
                constellation = _fps_subset(
                    source, config.primary_constellation_size, config.bits
                )
            elif condition == "learned" and selector is not None:
                constellation = selector(source, config.primary_constellation_size)
            else:
                raise ValueError("quick evaluation supports FPS or learned")
            reconstruction = decoder(constellation, num_output_points=config.num_points)
            losses = _chamfer_per_sample(reconstruction, target)
            total += float(losses.sum().item())
            count += len(target)
    return math.sqrt(total / count)


def _train_decoder(
    decoder: VariableConstellationDecoder,
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: BottleneckAuditConfig,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.decoder_learning_rate)
    history = []
    global_step = 0
    for epoch in range(1, config.decoder_epochs + 1):
        decoder.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            target = batch["points"].to(device)
            normals = batch["normals"].to(device)
            input_size = config.input_sizes[global_step % len(config.input_sizes)]
            constellation_size = config.constellation_sizes[
                global_step % len(config.constellation_sizes)
            ]
            sampler = config.training_samplers[
                (global_step // len(config.input_sizes)) % len(config.training_samplers)
            ]
            source, source_normals = _training_source(target, normals, input_size)
            constellation = _sample_subset(
                source,
                source_normals,
                constellation_size,
                config.bits,
                sampler,
            )
            constellation = quantize_ste(constellation, config.bits, training=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = decoder(constellation, num_output_points=config.num_points)
            loss = chamfer_squared(reconstruction, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
            count += len(target)
            global_step += 1

        validation_rmse = _quick_chamfer(
            decoder,
            None,
            validation_loader,
            device,
            config,
            condition="fps",
        )
        record = {
            "epoch": epoch,
            "training_chamfer_rmse": math.sqrt(total_loss / count),
            "validation_fps_chamfer_rmse": validation_rmse,
        }
        history.append(record)
        print(json.dumps({"stage": "decoder", **record}))
    return history


def _train_selector(
    selector: ProgressiveSubsetEncoder,
    decoder: VariableConstellationDecoder,
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: BottleneckAuditConfig,
) -> list[dict[str, Any]]:
    decoder.eval()
    decoder.requires_grad_(False)
    optimizer = torch.optim.Adam(
        selector.parameters(), lr=config.selector_learning_rate
    )
    history = []
    global_step = 0
    best_validation_rmse = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, config.selector_epochs + 1):
        selector.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            target = batch["points"].to(device)
            normals = batch["normals"].to(device)
            input_size = config.input_sizes[global_step % len(config.input_sizes)]
            constellation_size = config.constellation_sizes[
                global_step % len(config.constellation_sizes)
            ]
            source, _ = _training_source(target, normals, input_size)
            optimizer.zero_grad(set_to_none=True)
            constellation = selector(source, constellation_size)
            reconstruction = decoder(constellation, num_output_points=config.num_points)
            loss = chamfer_squared(reconstruction, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
            count += len(target)
            global_step += 1

        validation_rmse = _quick_chamfer(
            decoder,
            selector,
            validation_loader,
            device,
            config,
            condition="learned",
        )
        record = {
            "epoch": epoch,
            "training_chamfer_rmse": math.sqrt(total_loss / count),
            "validation_learned_chamfer_rmse": validation_rmse,
        }
        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in selector.state_dict().items()
            }
        history.append(record)
        print(json.dumps({"stage": "selector", **record}))
    if best_state is None:
        raise RuntimeError("selector training did not produce a checkpoint")
    selector.load_state_dict(best_state)
    for record in history:
        record["selected"] = record["epoch"] == best_epoch
    return history


def _rate_curve_check(
    runs: list[dict[str, Any]],
    *,
    split: str,
    primary_input_size: int,
    min_endpoint_improvement_percent: float,
    max_adjacent_regression_percent: float,
) -> dict[str, Any]:
    curve = sorted(
        (
            run
            for run in runs
            if run["condition"] == "fps" and run["input_size"] == primary_input_size
        ),
        key=lambda run: run["constellation_size"],
    )
    if len(curve) < 2:
        raise ValueError("at least two FPS rate points are required")
    values = [run[split]["chamfer_rmse"] for run in curve]
    endpoint = 100.0 * (values[0] - values[-1]) / values[0]
    adjacent = [
        100.0 * (current - previous) / previous
        for previous, current in zip(values[:-1], values[1:], strict=True)
    ]
    largest_regression = max(adjacent)
    return {
        "passed": endpoint >= min_endpoint_improvement_percent
        and largest_regression <= max_adjacent_regression_percent,
        "endpoint_improvement_percent": endpoint,
        "largest_adjacent_regression_percent": largest_regression,
        "constellation_sizes": [run["constellation_size"] for run in curve],
        "chamfer_rmse": values,
    }


def bottleneck_audit_gate(
    runs: list[dict[str, Any]],
    *,
    primary_input_size: int,
    primary_constellation_size: int,
    min_endpoint_improvement_percent: float,
    max_adjacent_regression_percent: float,
    max_learned_gap_vs_fps_percent: float,
    free_coordinate_headroom_percent: float,
    decoder_unchanged: bool,
) -> dict[str, Any]:
    validation_curve = _rate_curve_check(
        runs,
        split="validation",
        primary_input_size=primary_input_size,
        min_endpoint_improvement_percent=min_endpoint_improvement_percent,
        max_adjacent_regression_percent=max_adjacent_regression_percent,
    )
    ood_curve = _rate_curve_check(
        runs,
        split="parameter_ood",
        primary_input_size=primary_input_size,
        min_endpoint_improvement_percent=min_endpoint_improvement_percent,
        max_adjacent_regression_percent=max_adjacent_regression_percent,
    )

    def find(condition: str) -> dict[str, Any]:
        matches = [
            run
            for run in runs
            if run["condition"] == condition
            and run["input_size"] == primary_input_size
            and run["constellation_size"] == primary_constellation_size
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one primary {condition} run")
        return matches[0]

    fps = find("fps")
    learned = find("learned")
    best_subset = find("best_subset")
    free_coordinates = find("free_coordinates")

    def gap(split: str) -> float:
        return (
            100.0
            * (learned[split]["chamfer_rmse"] - fps[split]["chamfer_rmse"])
            / fps[split]["chamfer_rmse"]
        )

    validation_gap = gap("validation")
    ood_gap = gap("parameter_ood")
    validation_headroom = (
        100.0
        * (
            best_subset["validation"]["chamfer_rmse"]
            - free_coordinates["validation"]["chamfer_rmse"]
        )
        / best_subset["validation"]["chamfer_rmse"]
    )
    ood_headroom = (
        100.0
        * (
            best_subset["parameter_ood"]["chamfer_rmse"]
            - free_coordinates["parameter_ood"]["chamfer_rmse"]
        )
        / best_subset["parameter_ood"]["chamfer_rmse"]
    )
    checks = {
        "validation_fps_rate_curve": validation_curve["passed"],
        "parameter_ood_fps_rate_curve": ood_curve["passed"],
        "validation_learned_gap": validation_gap <= max_learned_gap_vs_fps_percent,
        "parameter_ood_learned_gap": ood_gap <= max_learned_gap_vs_fps_percent,
        "decoder_unchanged": decoder_unchanged,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "validation_fps_curve": validation_curve,
        "parameter_ood_fps_curve": ood_curve,
        "validation_learned_gap_vs_fps_percent": validation_gap,
        "parameter_ood_learned_gap_vs_fps_percent": ood_gap,
        "validation_free_coordinate_headroom_percent": validation_headroom,
        "parameter_ood_free_coordinate_headroom_percent": ood_headroom,
        "free_coordinate_headroom_detected": validation_headroom
        >= free_coordinate_headroom_percent
        and ood_headroom >= free_coordinate_headroom_percent,
        "max_learned_gap_vs_fps_percent": max_learned_gap_vs_fps_percent,
        "free_coordinate_headroom_threshold_percent": (
            free_coordinate_headroom_percent
        ),
    }


def bottleneck_audit(
    config: BottleneckAuditConfig,
    *,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, parameter_ood_loader = _make_loaders(config)
    decoder = VariableConstellationDecoder(
        config.num_points,
        max(config.constellation_sizes),
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    selector = ProgressiveSubsetEncoder(
        max(config.constellation_sizes),
        bits=config.bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        selection_temperature=config.selection_temperature,
    ).to(device)
    decoder_path = output_dir / "decoder.pt"
    selector_path = output_dir / "selector.pt"
    started = time.perf_counter()

    if resume and decoder_path.exists():
        decoder_checkpoint = torch.load(
            decoder_path, map_location=device, weights_only=True
        )
        decoder.load_state_dict(decoder_checkpoint["model"])
        decoder_history = decoder_checkpoint.get("history", [])
    else:
        decoder_history = _train_decoder(
            decoder, train_loader, validation_loader, device, config
        )
        torch.save(
            {"model": decoder.state_dict(), "history": decoder_history},
            decoder_path,
        )

    decoder_before = {
        name: value.detach().cpu().clone()
        for name, value in decoder.state_dict().items()
    }
    selector_train_loader, _, _ = _make_loaders(config)
    set_seed(config.seed + 1)
    if resume and selector_path.exists():
        selector_checkpoint = torch.load(
            selector_path, map_location=device, weights_only=True
        )
        selector.load_state_dict(selector_checkpoint["model"])
        selector_history = selector_checkpoint.get("history", [])
        decoder.requires_grad_(False)
    else:
        selector_history = _train_selector(
            selector,
            decoder,
            selector_train_loader,
            validation_loader,
            device,
            config,
        )
        torch.save(
            {"model": selector.state_dict(), "history": selector_history},
            selector_path,
        )
    decoder_unchanged = all(
        torch.equal(decoder_before[name], value.detach().cpu())
        for name, value in decoder.state_dict().items()
    )

    operating_points = {
        (config.primary_input_size, constellation_size)
        for constellation_size in config.constellation_sizes
    }
    operating_points.update(
        (input_size, config.primary_constellation_size)
        for input_size in config.input_sizes
    )
    runs = []
    total_runs = len(operating_points) * len(CONDITIONS) + 2
    run_index = 0
    for input_size, constellation_size in sorted(operating_points):
        for condition_index, condition in enumerate(CONDITIONS):
            run_index += 1
            print(
                json.dumps(
                    {
                        "evaluation_point": run_index,
                        "total_points": total_runs,
                        "condition": condition,
                        "input_size": input_size,
                        "constellation_size": constellation_size,
                    }
                )
            )
            validation = _evaluate_condition(
                decoder,
                selector,
                validation_loader,
                device,
                config,
                condition=condition,
                input_size=input_size,
                constellation_size=constellation_size,
                seed_offset=(
                    1000 + 100 * input_size + 10 * constellation_size + condition_index
                ),
            )
            parameter_ood = _evaluate_condition(
                decoder,
                selector,
                parameter_ood_loader,
                device,
                config,
                condition=condition,
                input_size=input_size,
                constellation_size=constellation_size,
                seed_offset=(
                    2000 + 100 * input_size + 10 * constellation_size + condition_index
                ),
            )
            runs.append(
                {
                    "condition": condition,
                    "input_size": input_size,
                    "constellation_size": constellation_size,
                    "coordinate_payload_bits": 3 * constellation_size * config.bits,
                    "bits_per_input_point": 3
                    * constellation_size
                    * config.bits
                    / input_size,
                    "validation": validation,
                    "parameter_ood": parameter_ood,
                }
            )

    for oracle_index, condition in enumerate(("best_subset", "free_coordinates")):
        run_index += 1
        print(
            json.dumps(
                {
                    "evaluation_point": run_index,
                    "total_points": total_runs,
                    "condition": condition,
                    "input_size": config.primary_input_size,
                    "constellation_size": config.primary_constellation_size,
                }
            )
        )
        validation = _evaluate_condition(
            decoder,
            selector,
            validation_loader,
            device,
            config,
            condition=condition,
            input_size=config.primary_input_size,
            constellation_size=config.primary_constellation_size,
            seed_offset=3000 + oracle_index,
        )
        parameter_ood = _evaluate_condition(
            decoder,
            selector,
            parameter_ood_loader,
            device,
            config,
            condition=condition,
            input_size=config.primary_input_size,
            constellation_size=config.primary_constellation_size,
            seed_offset=4000 + oracle_index,
        )
        runs.append(
            {
                "condition": condition,
                "input_size": config.primary_input_size,
                "constellation_size": config.primary_constellation_size,
                "coordinate_payload_bits": 3
                * config.primary_constellation_size
                * config.bits,
                "bits_per_input_point": 3
                * config.primary_constellation_size
                * config.bits
                / config.primary_input_size,
                "validation": validation,
                "parameter_ood": parameter_ood,
            }
        )

    gate = bottleneck_audit_gate(
        runs,
        primary_input_size=config.primary_input_size,
        primary_constellation_size=config.primary_constellation_size,
        min_endpoint_improvement_percent=(config.min_endpoint_improvement_percent),
        max_adjacent_regression_percent=config.max_adjacent_regression_percent,
        max_learned_gap_vs_fps_percent=(config.max_learned_gap_vs_fps_percent),
        free_coordinate_headroom_percent=config.free_coordinate_headroom_percent,
        decoder_unchanged=decoder_unchanged,
    )
    result = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "decoder_parameter_count": sum(
            parameter.numel() for parameter in decoder.parameters()
        ),
        "selector_parameter_count": sum(
            parameter.numel() for parameter in selector.parameters()
        ),
        "decoder_unchanged_during_selector_training": decoder_unchanged,
        "elapsed_seconds": time.perf_counter() - started,
        "decoder_history": decoder_history,
        "selector_history": selector_history,
        "runs": runs,
        "gate": gate,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_004_frozen_decoder.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = bottleneck_audit(
        BottleneckAuditConfig.from_json(args.config),
        device_name=args.device,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("decoder_history", "selector_history", "runs")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
