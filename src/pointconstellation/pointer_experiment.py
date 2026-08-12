"""Experiment 012: autoregressive residual-aware pointer subset selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pointconstellation.data import ProceduralPointCloudDataset
from pointconstellation.losses import chamfer_squared, pairwise_squared
from pointconstellation.models.bottleneck import (
    ProgressiveSubsetEncoder,
    VariableConstellationDecoder,
)
from pointconstellation.models.pointer import (
    AutoregressivePointerSubsetEncoder,
    PointerTrace,
)
from pointconstellation.quantization import quantize_coordinates, quantize_ste
from pointconstellation.train import select_device, set_seed


@dataclass(frozen=True)
class PointerExperimentConfig:
    num_points: int = 32
    input_sizes: tuple[int, ...] = (16, 32)
    constellation_sizes: tuple[int, ...] = (4, 8)
    bits: int = 10
    train_samples: int = 14
    validation_samples: int = 7
    parameter_ood_samples: int = 7
    batch_size: int = 7
    decoder_epochs: int = 1
    selector_epochs: int = 1
    decoder_learning_rate: float = 1e-3
    selector_learning_rate: float = 1e-3
    coverage_weight: float = 0.05
    entropy_weight: float = 0.001
    feature_width: int = 24
    num_heads: int = 4
    num_layers: int = 1
    selection_temperature: float = 0.2
    stochastic_training: bool = True
    beam_width: int = 2
    beam_branch_factor: int = 2
    beam_score_partial_candidates: bool = False
    decoder_checkpoint: str | None = None
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_012_pointer_smoke"

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
                "constellation sizes must fit within the smallest input size"
            )
        if not 2 <= self.bits <= 24:
            raise ValueError("bits must be between 2 and 24")
        if (
            min(
                self.train_samples,
                self.validation_samples,
                self.parameter_ood_samples,
                self.batch_size,
                self.decoder_epochs,
                self.selector_epochs,
            )
            < 1
        ):
            raise ValueError("sample counts, batch size, and epochs must be positive")
        if min(self.decoder_learning_rate, self.selector_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if self.coverage_weight < 0 or self.entropy_weight < 0:
            raise ValueError("coverage and entropy weights cannot be negative")
        if self.feature_width < 4 or self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        if self.beam_width < 1 or self.beam_branch_factor < 1:
            raise ValueError("beam width and branch factor must be positive")

    @classmethod
    def from_json(cls, path: Path) -> PointerExperimentConfig:
        values = json.loads(path.read_text())
        for key in ("input_sizes", "constellation_sizes"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def _make_loaders(
    config: PointerExperimentConfig,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
]:
    datasets = {
        split: ProceduralPointCloudDataset(
            count,
            num_points=config.num_points,
            seed=config.seed,
            split=split,
        )
        for split, count in (
            ("train", config.train_samples),
            ("validation", config.validation_samples),
            ("parameter_ood", config.parameter_ood_samples),
        )
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


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _named_state_hash(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _fps(points: Tensor, constellation_size: int, bits: int) -> Tensor:
    batch_size, num_points, _ = points.shape
    rows = torch.arange(batch_size, device=points.device)
    farthest = ((points - points.mean(dim=1, keepdim=True)) ** 2).sum(-1).argmax(1)
    minimum = torch.full(
        (batch_size, num_points),
        torch.inf,
        dtype=points.dtype,
        device=points.device,
    )
    selected = torch.empty(
        (batch_size, constellation_size), dtype=torch.long, device=points.device
    )
    for index in range(constellation_size):
        selected[:, index] = farthest
        anchor = points[rows, farthest][:, None]
        minimum = torch.minimum(minimum, ((points - anchor) ** 2).sum(-1))
        farthest = minimum.argmax(1)
    anchors = points.gather(1, selected[:, :, None].expand(-1, -1, 3))
    return quantize_coordinates(anchors, bits)


def _random_subset(points: Tensor, constellation_size: int, bits: int) -> Tensor:
    scores = torch.rand(points.shape[:2], device=points.device)
    indices = scores.topk(constellation_size, dim=1).indices
    anchors = points.gather(1, indices[:, :, None].expand(-1, -1, 3))
    return quantize_coordinates(anchors, bits)


def _operating_points(config: PointerExperimentConfig) -> tuple[tuple[int, int], ...]:
    """Return the complete variable-rate training grid."""

    return tuple(
        (input_size, constellation_size)
        for input_size in config.input_sizes
        for constellation_size in config.constellation_sizes
    )


def _train_decoder(
    decoder: VariableConstellationDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: PointerExperimentConfig,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.decoder_learning_rate)
    history: list[dict[str, float | int]] = []
    global_step = 0
    for epoch in range(1, config.decoder_epochs + 1):
        decoder.train()
        total = 0.0
        count = 0
        for batch in loader:
            target = batch["points"].to(device)
            for input_size, constellation_size in _operating_points(config):
                source = target[:, :input_size]
                if global_step % 2:
                    constellation = _random_subset(
                        source, constellation_size, config.bits
                    )
                else:
                    constellation = _fps(source, constellation_size, config.bits)
                constellation = quantize_ste(
                    constellation, config.bits, training=True, jitter=True
                )
                optimizer.zero_grad(set_to_none=True)
                reconstruction = decoder(
                    constellation, num_output_points=config.num_points
                )
                loss = chamfer_squared(reconstruction, target)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(target)
                count += len(target)
                global_step += 1
        record: dict[str, float | int] = {
            "epoch": epoch,
            "training_chamfer_rmse": math.sqrt(total / count),
        }
        history.append(record)
        print(json.dumps({"stage": "decoder", **record}))
    return history


def _copy_matched_scalar_initialization(
    pointer: AutoregressivePointerSubsetEncoder,
    scalar: ProgressiveSubsetEncoder,
) -> tuple[str, str]:
    scalar.point_embedding.load_state_dict(pointer.point_embedding.state_dict())
    scalar.point_relations.load_state_dict(pointer.point_relations.state_dict())
    scalar.score_head.load_state_dict(pointer.score_head.state_dict())
    pointer_common = {
        f"point_embedding.{name}": value
        for name, value in pointer.point_embedding.state_dict().items()
    }
    pointer_common.update(
        {
            f"point_relations.{name}": value
            for name, value in pointer.point_relations.state_dict().items()
        }
    )
    pointer_common.update(
        {
            f"score_head.{name}": value
            for name, value in pointer.score_head.state_dict().items()
        }
    )
    scalar_common = {
        name: value
        for name, value in scalar.state_dict().items()
        if name.startswith(("point_embedding.", "point_relations.", "score_head."))
    }
    return _named_state_hash(pointer_common), _named_state_hash(scalar_common)


def _coverage_loss(constellation: Tensor, target: Tensor) -> Tensor:
    return pairwise_squared(constellation, target).amin(dim=1).mean()


def _normalized_entropy(trace: PointerTrace, num_input_points: int) -> Tensor:
    denominators = [
        math.log(max(num_input_points - turn, 2))
        for turn in range(trace.entropies.shape[1])
    ]
    scale = trace.entropies.new_tensor(denominators)[None]
    return (trace.entropies / scale).mean()


def _train_selectors(
    pointer: AutoregressivePointerSubsetEncoder,
    scalar: ProgressiveSubsetEncoder,
    decoder: VariableConstellationDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: PointerExperimentConfig,
) -> tuple[list[dict[str, Any]], int, int]:
    decoder.eval().requires_grad_(False)
    pointer_optimizer = torch.optim.Adam(
        pointer.parameters(), lr=config.selector_learning_rate
    )
    scalar_optimizer = torch.optim.Adam(
        scalar.parameters(), lr=config.selector_learning_rate
    )
    history: list[dict[str, Any]] = []
    pointer_updates = 0
    scalar_updates = 0
    global_step = 0
    for epoch in range(1, config.selector_epochs + 1):
        pointer.train()
        scalar.train()
        totals = {
            "pointer": {"loss": 0.0, "chamfer": 0.0, "coverage": 0.0},
            "scalar": {"loss": 0.0, "chamfer": 0.0, "coverage": 0.0},
        }
        entropy_total = 0.0
        count = 0
        for batch in loader:
            target = batch["points"].to(device)
            for input_size, constellation_size in _operating_points(config):
                source = target[:, :input_size]

                pointer_optimizer.zero_grad(set_to_none=True)
                pointer_output = pointer(source, constellation_size, return_trace=True)
                assert isinstance(pointer_output, tuple)
                pointer_constellation, trace = pointer_output
                pointer_reconstruction = decoder(
                    pointer_constellation, num_output_points=config.num_points
                )
                pointer_chamfer = chamfer_squared(pointer_reconstruction, target)
                pointer_coverage = _coverage_loss(pointer_constellation, target)
                normalized_entropy = _normalized_entropy(trace, input_size)
                pointer_loss = (
                    pointer_chamfer
                    + config.coverage_weight * pointer_coverage
                    - config.entropy_weight * normalized_entropy
                )
                pointer_loss.backward()
                pointer_optimizer.step()
                pointer_updates += 1

                scalar_optimizer.zero_grad(set_to_none=True)
                scalar_constellation = scalar(source, constellation_size)
                scalar_reconstruction = decoder(
                    scalar_constellation, num_output_points=config.num_points
                )
                scalar_chamfer = chamfer_squared(scalar_reconstruction, target)
                scalar_coverage = _coverage_loss(scalar_constellation, target)
                scalar_loss = scalar_chamfer + config.coverage_weight * scalar_coverage
                scalar_loss.backward()
                scalar_optimizer.step()
                scalar_updates += 1

                batch_size = len(target)
                for name, value in (
                    ("loss", pointer_loss),
                    ("chamfer", pointer_chamfer),
                    ("coverage", pointer_coverage),
                ):
                    totals["pointer"][name] += float(value.item()) * batch_size
                for name, value in (
                    ("loss", scalar_loss),
                    ("chamfer", scalar_chamfer),
                    ("coverage", scalar_coverage),
                ):
                    totals["scalar"][name] += float(value.item()) * batch_size
                entropy_total += float(normalized_entropy.item()) * batch_size
                count += batch_size
                global_step += 1

        record: dict[str, Any] = {
            "epoch": epoch,
            "pointer_normalized_selection_entropy": entropy_total / count,
        }
        for method in ("pointer", "scalar"):
            means = {name: value / count for name, value in totals[method].items()}
            record[method] = {
                **means,
                "chamfer_rmse": math.sqrt(means["chamfer"]),
                "coverage_rmse": math.sqrt(means["coverage"]),
            }
        history.append(record)
        print(json.dumps({"stage": "selectors", **record}))
    return history, pointer_updates, scalar_updates


def _sample_metrics(
    reconstruction: Tensor,
    target: Tensor,
    constellation: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    reconstruction_distances = pairwise_squared(reconstruction, target)
    chamfer = 0.5 * (
        reconstruction_distances.amin(dim=2).mean(dim=1)
        + reconstruction_distances.amin(dim=1).mean(dim=1)
    )
    anchor_distances = pairwise_squared(constellation, target)
    surface = anchor_distances.amin(dim=2).mean(dim=1)
    coverage = anchor_distances.amin(dim=1).mean(dim=1)
    return chamfer, surface, coverage


def _unique_counts(constellation: Tensor) -> list[int]:
    return [int(torch.unique(sample, dim=0).shape[0]) for sample in constellation]


def _empty_method_totals() -> dict[str, float | int]:
    return {
        "chamfer": 0.0,
        "surface": 0.0,
        "coverage": 0.0,
        "unique_sum": 0,
        "unique_min": 1 << 30,
        "fully_unique": 0,
        "entropy_sum": 0.0,
    }


def _evaluate_request(
    pointer: AutoregressivePointerSubsetEncoder,
    scalar: ProgressiveSubsetEncoder,
    decoder: VariableConstellationDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: PointerExperimentConfig,
    *,
    input_size: int,
    constellation_size: int,
) -> dict[str, Any]:
    pointer.eval()
    scalar.eval()
    decoder.eval()
    method_names = ["pointer_greedy", "scalar", "fps"]
    if config.beam_width > 1:
        method_names.insert(1, "pointer_beam")
    totals = {method: _empty_method_totals() for method in method_names}
    beam_decoder_evaluations = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["points"].to(device)
            source = target[:, :input_size]
            pointer_output = pointer(source, constellation_size, return_trace=True)
            assert isinstance(pointer_output, tuple)
            greedy, trace = pointer_output
            constellations = {
                "pointer_greedy": greedy,
                "scalar": scalar(source, constellation_size),
                "fps": _fps(source, constellation_size, config.bits),
            }
            if config.beam_width > 1:
                beam = pointer.beam_search(
                    source,
                    constellation_size,
                    decoder=decoder,
                    target=target,
                    num_output_points=config.num_points,
                    beam_width=config.beam_width,
                    branch_factor=config.beam_branch_factor,
                    score_partial_candidates=config.beam_score_partial_candidates,
                )
                constellations["pointer_beam"] = beam.coordinates
                beam_decoder_evaluations += int(beam.decoder_evaluations.sum().item())

            for method, raw_constellation in constellations.items():
                constellation = quantize_coordinates(raw_constellation, config.bits)
                reconstruction = decoder(
                    constellation, num_output_points=config.num_points
                )
                chamfer, surface, coverage = _sample_metrics(
                    reconstruction, target, constellation
                )
                method_totals = totals[method]
                method_totals["chamfer"] += float(chamfer.sum().item())
                method_totals["surface"] += float(surface.sum().item())
                method_totals["coverage"] += float(coverage.sum().item())
                unique = _unique_counts(constellation)
                method_totals["unique_sum"] += sum(unique)
                method_totals["unique_min"] = min(
                    int(method_totals["unique_min"]), min(unique)
                )
                method_totals["fully_unique"] += sum(
                    value == constellation_size for value in unique
                )
            totals["pointer_greedy"]["entropy_sum"] += float(
                _normalized_entropy(trace, input_size).item()
            ) * len(target)
            count += len(target)

    results: dict[str, Any] = {}
    for method, values in totals.items():
        chamfer = float(values["chamfer"]) / count
        surface = float(values["surface"]) / count
        coverage = float(values["coverage"]) / count
        results[method] = {
            "exact_quantized": True,
            "chamfer": chamfer,
            "chamfer_rmse": math.sqrt(chamfer),
            "surface_proxy": surface,
            "surface_proxy_rmse": math.sqrt(surface),
            "coverage": coverage,
            "coverage_rmse": math.sqrt(coverage),
            "unique_k_mean": int(values["unique_sum"]) / count,
            "unique_k_min": int(values["unique_min"]),
            "fully_unique_fraction": int(values["fully_unique"]) / count,
        }
    results["pointer_greedy"]["normalized_selection_entropy"] = (
        float(totals["pointer_greedy"]["entropy_sum"]) / count
    )
    if "pointer_beam" in results:
        greedy_rmse = results["pointer_greedy"]["chamfer_rmse"]
        beam_rmse = results["pointer_beam"]["chamfer_rmse"]
        results["beam_gain"] = {
            "greedy_minus_beam_chamfer_rmse": greedy_rmse - beam_rmse,
            "improvement_percent": 100.0
            * (greedy_rmse - beam_rmse)
            / max(greedy_rmse, 1e-12),
        }
        results["pointer_beam"]["target_oracle_decoder_evaluations_per_cloud"] = (
            beam_decoder_evaluations / count
        )
    return results


def _cross_k_turnover(
    pointer: AutoregressivePointerSubsetEncoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: PointerExperimentConfig,
    *,
    input_size: int,
) -> list[dict[str, float | int]]:
    sizes = sorted(config.constellation_sizes)
    if len(sizes) < 2:
        return []
    totals = [0.0] * (len(sizes) - 1)
    count = 0
    pointer.eval()
    with torch.no_grad():
        for batch in loader:
            source = batch["points"].to(device)[:, :input_size]
            selected: dict[int, Tensor] = {}
            for size in sizes:
                output = pointer(source, size, return_trace=True)
                assert isinstance(output, tuple)
                selected[size] = output[1].indices
            for pair_index, (smaller, larger) in enumerate(
                zip(sizes, sizes[1:], strict=False)
            ):
                first = selected[smaller]
                second = selected[larger]
                matches = (first[:, :, None] == second[:, None, :]).any(dim=2)
                intersection = matches.sum(dim=1)
                turnover = 1.0 - intersection.float() / smaller
                totals[pair_index] += float(turnover.sum().item())
            count += len(source)
    return [
        {
            "smaller_k": smaller,
            "larger_k": larger,
            "selection_turnover": totals[index] / count,
        }
        for index, (smaller, larger) in enumerate(zip(sizes, sizes[1:], strict=False))
    ]


def run_pointer_experiment(
    config: PointerExperimentConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train the frozen-decoder pointer comparison and save all artifacts."""

    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, ood_loader = _make_loaders(config)
    decoder = VariableConstellationDecoder(
        config.num_points,
        max(config.constellation_sizes),
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    started = time.perf_counter()
    decoder_source = "trained"
    if config.decoder_checkpoint is None:
        decoder_history = _train_decoder(decoder, train_loader, device, config)
    else:
        checkpoint_path = Path(config.decoder_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"decoder checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state = checkpoint.get("model", checkpoint.get("decoder", checkpoint))
        decoder.load_state_dict(state)
        decoder_history = checkpoint.get("history", [])
        decoder_source = str(checkpoint_path)
    torch.save(
        {"model": decoder.state_dict(), "history": decoder_history},
        output_dir / "decoder.pt",
    )

    torch.manual_seed(config.seed + 12012)
    pointer = AutoregressivePointerSubsetEncoder(
        max(config.constellation_sizes),
        bits=config.bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        selection_temperature=config.selection_temperature,
        stochastic_training=config.stochastic_training,
    ).to(device)
    scalar = ProgressiveSubsetEncoder(
        max(config.constellation_sizes),
        bits=config.bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        selection_temperature=config.selection_temperature,
        stochastic_training=config.stochastic_training,
        quantization_jitter=False,
    ).to(device)
    pointer_common_hash, scalar_common_hash = _copy_matched_scalar_initialization(
        pointer, scalar
    )
    common_initialization_matched = pointer_common_hash == scalar_common_hash
    if not common_initialization_matched:
        raise RuntimeError("pointer/scalar common initialization does not match")

    decoder.eval().requires_grad_(False)
    decoder_hash_before = _state_hash(decoder)
    selector_history, pointer_updates, scalar_updates = _train_selectors(
        pointer, scalar, decoder, train_loader, device, config
    )
    decoder_hash_after_training = _state_hash(decoder)
    if pointer_updates != scalar_updates:
        raise RuntimeError("pointer and scalar update budgets do not match")

    evaluation: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "parameter_ood": [],
    }
    turnover: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "parameter_ood": [],
    }
    for split, loader in (
        ("validation", validation_loader),
        ("parameter_ood", ood_loader),
    ):
        for input_size in config.input_sizes:
            for constellation_size in config.constellation_sizes:
                evaluation[split].append(
                    {
                        "input_size": input_size,
                        "constellation_size": constellation_size,
                        "coordinate_payload_bits": (
                            3 * constellation_size * config.bits
                        ),
                        "methods": _evaluate_request(
                            pointer,
                            scalar,
                            decoder,
                            loader,
                            device,
                            config,
                            input_size=input_size,
                            constellation_size=constellation_size,
                        ),
                    }
                )
            turnover[split].append(
                {
                    "input_size": input_size,
                    "adjacent_requests": _cross_k_turnover(
                        pointer,
                        loader,
                        device,
                        config,
                        input_size=input_size,
                    ),
                }
            )

    decoder_hash_final = _state_hash(decoder)
    decoder_unchanged = (
        decoder_hash_before == decoder_hash_after_training == decoder_hash_final
    )
    if not decoder_unchanged:
        raise RuntimeError("frozen decoder changed during selector work")

    result = {
        "experiment": "012_autoregressive_pointer",
        "config": asdict(config),
        "device": str(device),
        "decoder_source": decoder_source,
        "decoder_history": decoder_history,
        "decoder_hash_before_selectors": decoder_hash_before,
        "decoder_hash_after_training": decoder_hash_after_training,
        "decoder_hash_final": decoder_hash_final,
        "decoder_unchanged": decoder_unchanged,
        "pointer_common_initial_hash": pointer_common_hash,
        "scalar_common_initial_hash": scalar_common_hash,
        "common_initialization_matched": common_initialization_matched,
        "pointer_updates": pointer_updates,
        "scalar_updates": scalar_updates,
        "update_budget_matched": pointer_updates == scalar_updates,
        "training_operating_points": [
            {"input_size": input_size, "constellation_size": constellation_size}
            for input_size, constellation_size in _operating_points(config)
        ],
        "pointer_scalar_stochasticity_matched": True,
        "pointer_scalar_quantization_jitter_matched": True,
        "pointer_only_entropy_bonus_active": config.entropy_weight != 0,
        "decoder_parameter_count": sum(
            parameter.numel() for parameter in decoder.parameters()
        ),
        "pointer_parameter_count": sum(
            parameter.numel() for parameter in pointer.parameters()
        ),
        "scalar_parameter_count": sum(
            parameter.numel() for parameter in scalar.parameters()
        ),
        "selector_history": selector_history,
        "evaluation": evaluation,
        "cross_k_turnover": turnover,
        "elapsed_seconds": time.perf_counter() - started,
        "caveats": [
            "Surface proxy is distance to observed samples, not an analytic surface.",
            "Common point/context/scalar-score parameters are matched; the pointer "
            "also has conditional-state parameters unavailable to scalar ranking.",
            "Beam scoring uses the evaluation target and is an oracle diagnostic, "
            "not a deployable encoder unless a target-derived score is available.",
            "The smoke config disables the pointer-only entropy bonus and matches "
            "stochastic selection and quantizer jitter across pointer and scalar.",
            "Cross-K turnover measures independent K-conditioned requests and is "
            "not itself evidence of better reconstruction.",
        ],
    }
    torch.save(
        {"model": pointer.state_dict(), "config": asdict(config)},
        output_dir / "pointer.pt",
    )
    torch.save(
        {"model": scalar.state_dict(), "config": asdict(config)},
        output_dir / "scalar.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_012_pointer_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--decoder-checkpoint")
    args = parser.parse_args()

    config = PointerExperimentConfig.from_json(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.decoder_checkpoint is not None:
        config = replace(config, decoder_checkpoint=args.decoder_checkpoint)
    result = run_pointer_experiment(config, device_name=args.device)
    print(
        json.dumps(
            {
                "device": result["device"],
                "decoder_unchanged": result["decoder_unchanged"],
                "common_initialization_matched": result[
                    "common_initialization_matched"
                ],
                "update_budget_matched": result["update_budget_matched"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
