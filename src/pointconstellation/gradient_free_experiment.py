"""Experiment 011: compare decoder-scored gradient-free constellation search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from pointconstellation.data import generate_sample
from pointconstellation.losses import chamfer_squared, pairwise_squared
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.gradient_free import (
    SearchResult,
    adam_ste_search,
    coordinate_cem_search,
    subset_mutation_search,
)
from pointconstellation.quantization import (
    quantization_step,
    quantize_coordinates,
)
from pointconstellation.train import select_device, set_seed

START_KINDS = ("fps", "random")
METHODS = ("adam_ste", "coordinate_cem", "subset_mutation")


@dataclass(frozen=True)
class GradientFreeExperimentConfig:
    """Validated fixed-rate search comparison configuration."""

    num_points: int = 16
    input_size: int = 16
    constellation_size: int = 4
    bits: int = 8
    train_samples: int = 8
    validation_samples: int = 3
    parameter_ood_samples: int = 3
    batch_size: int = 4
    decoder_epochs: int = 1
    decoder_learning_rate: float = 1e-3
    feature_width: int = 16
    num_heads: int = 4
    num_layers: int = 1
    start_kinds: tuple[str, ...] = START_KINDS
    population_size: int = 4
    generations: int = 2
    cem_elite_fraction: float = 0.5
    cem_initial_std: float = 0.15
    cem_minimum_std: float = 0.01
    mutation_swaps: int = 1
    adam_learning_rate: float = 0.04
    perturbation_samples: int = 3
    decoder_checkpoint: str | None = None
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_011_gradient_free_smoke"

    def __post_init__(self) -> None:
        if self.num_points < 8:
            raise ValueError("num_points must be at least 8")
        if not 8 <= self.input_size <= self.num_points:
            raise ValueError("input_size must be between 8 and num_points")
        if not 2 <= self.constellation_size <= self.input_size:
            raise ValueError("constellation_size must fit within input_size")
        if not 2 <= self.bits <= 24:
            raise ValueError("bits must be between 2 and 24")
        if (
            min(
                self.train_samples,
                self.validation_samples,
                self.parameter_ood_samples,
                self.batch_size,
                self.decoder_epochs,
                self.population_size,
                self.generations,
                self.mutation_swaps,
                self.perturbation_samples,
                self.num_layers,
            )
            < 1
        ):
            raise ValueError("sample, epoch, search, and layer counts must be positive")
        if self.population_size < 2:
            raise ValueError("population_size must be at least two")
        if self.feature_width < 4 or self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if min(self.decoder_learning_rate, self.adam_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 < self.cem_elite_fraction <= 1:
            raise ValueError("cem_elite_fraction must be in (0, 1]")
        if min(self.cem_initial_std, self.cem_minimum_std) <= 0:
            raise ValueError("CEM standard deviations must be positive")
        if not self.start_kinds or len(set(self.start_kinds)) != len(self.start_kinds):
            raise ValueError("start_kinds must be nonempty and unique")
        unknown = set(self.start_kinds) - set(START_KINDS)
        if unknown:
            raise ValueError(f"unknown start kinds: {sorted(unknown)}")

    @property
    def search_budget(self) -> int:
        return 1 + self.population_size * self.generations

    @classmethod
    def from_json(cls, path: Path) -> GradientFreeExperimentConfig:
        values = json.loads(path.read_text())
        if "start_kinds" in values:
            values["start_kinds"] = tuple(values["start_kinds"])
        return cls(**values)


def _load_points(
    count: int,
    config: GradientFreeExperimentConfig,
    split: str,
    device: torch.device,
) -> Tensor:
    return torch.stack(
        [
            torch.from_numpy(
                generate_sample(
                    sample_id,
                    num_points=config.num_points,
                    seed=config.seed,
                    split=split,
                ).points
            )
            for sample_id in range(count)
        ]
    ).to(device)


def _chamfer_per_sample(first: Tensor, second: Tensor) -> Tensor:
    distances = pairwise_squared(first, second)
    return 0.5 * (distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1))


def farthest_point_indices(points: Tensor, constellation_size: int) -> Tensor:
    """Return deterministic FPS indices for every batch item."""

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
    return selected


def _random_subset_indices(
    batch_size: int,
    num_points: int,
    constellation_size: int,
    *,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randperm(num_points, generator=generator)[:constellation_size]
            for _ in range(batch_size)
        ]
    )
    return indices.to(device)


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_hash(tensor: Tensor) -> str:
    digest = hashlib.sha256()
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _train_decoder(
    decoder: VariableConstellationDecoder,
    points: Tensor,
    config: GradientFreeExperimentConfig,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.decoder_learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 11_000)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.decoder_epochs + 1):
        decoder.train()
        order = torch.randperm(len(points), generator=generator).to(points.device)
        total = 0.0
        count = 0
        for start in range(0, len(points), config.batch_size):
            target = points[order[start : start + config.batch_size]]
            source = target[:, : config.input_size]
            indices = farthest_point_indices(source, config.constellation_size)
            constellation = quantize_coordinates(
                source.gather(1, indices[:, :, None].expand(-1, -1, 3)),
                config.bits,
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction = decoder(constellation, num_output_points=config.num_points)
            loss = chamfer_squared(reconstruction, target)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(target)
            count += len(target)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "training_chamfer_rmse": math.sqrt(total / count),
        }
        history.append(record)
        print(json.dumps({"stage": "decoder", **record}))
    return history


def _decoder_scorer(
    decoder: VariableConstellationDecoder,
    target: Tensor,
    num_output_points: int,
) -> Callable[[Tensor], Tensor]:
    """Build a search score containing only frozen-decoder distortion."""

    def score(coordinates: Tensor) -> Tensor:
        if coordinates.ndim == 3:
            reconstruction = decoder(coordinates, num_output_points=num_output_points)
            return _chamfer_per_sample(reconstruction, target)
        if coordinates.ndim == 4:
            batch_size, population, constellation_size, _ = coordinates.shape
            flat = coordinates.reshape(batch_size * population, constellation_size, 3)
            reconstruction = decoder(flat, num_output_points=num_output_points)
            expanded_target = (
                target[:, None]
                .expand(-1, population, -1, -1)
                .reshape(batch_size * population, target.shape[1], 3)
            )
            return _chamfer_per_sample(reconstruction, expanded_target).reshape(
                batch_size, population
            )
        raise ValueError("coordinates must have shape (B, K, 3) or (B, P, K, 3)")

    return score


def _curve_json(result: SearchResult) -> list[dict[str, float | int]]:
    records = []
    for index, evaluations in enumerate(result.evaluation_counts):
        mean_loss = float(result.best_loss_history[index].mean().item())
        records.append(
            {
                "decoder_evaluations_per_cloud": evaluations,
                "mean_best_chamfer": mean_loss,
                "mean_best_chamfer_rmse": math.sqrt(mean_loss),
            }
        )
    return records


def _diagnostics(
    result: SearchResult,
    target: Tensor,
    score: Callable[[Tensor], Tensor],
    config: GradientFreeExperimentConfig,
    *,
    seed: int,
) -> dict[str, float | int]:
    constellation = result.coordinates
    anchor_distances = pairwise_squared(constellation, target)
    surface = anchor_distances.amin(dim=2).mean(dim=1)
    coverage = anchor_distances.amin(dim=1).mean(dim=1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randint(
        0,
        2,
        (
            len(constellation),
            config.perturbation_samples,
            constellation.shape[1],
            3,
        ),
        generator=generator,
    ).to(constellation.device)
    directions = directions.to(constellation.dtype).mul_(2).sub_(1)
    perturbed = quantize_coordinates(
        constellation[:, None] + quantization_step(config.bits) * directions,
        config.bits,
    )
    with torch.no_grad():
        perturbed_losses = score(perturbed)
    changes = perturbed_losses - result.losses[:, None]
    surface_mean = float(surface.mean().item())
    coverage_mean = float(coverage.mean().item())
    return {
        "mean_chamfer": float(result.losses.mean().item()),
        "chamfer_rmse": math.sqrt(float(result.losses.mean().item())),
        "surface_proxy": surface_mean,
        "surface_proxy_rmse": math.sqrt(surface_mean),
        "coverage": coverage_mean,
        "coverage_rmse": math.sqrt(coverage_mean),
        "mean_perturbation_chamfer_change": float(changes.mean().item()),
        "perturbation_sensitivity": math.sqrt(float(changes.square().mean().item())),
        "diagnostic_decoder_evaluations_per_cloud": config.perturbation_samples,
    }


def _evaluate_start(
    decoder: VariableConstellationDecoder,
    points: Tensor,
    config: GradientFreeExperimentConfig,
    *,
    split: str,
    start_kind: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    target = points
    source = target[:, : config.input_size]
    if start_kind == "fps":
        initial_indices = farthest_point_indices(source, config.constellation_size)
    elif start_kind == "random":
        initial_indices = _random_subset_indices(
            len(source),
            source.shape[1],
            config.constellation_size,
            seed=seed,
            device=source.device,
        )
    else:
        raise ValueError(f"unknown start kind: {start_kind}")
    quantized_source = quantize_coordinates(source, config.bits)
    initial = quantized_source.gather(1, initial_indices[:, :, None].expand(-1, -1, 3))
    score = _decoder_scorer(decoder, target, config.num_points)
    results = {
        "adam_ste": adam_ste_search(
            score,
            initial,
            bits=config.bits,
            decoder_evaluation_budget=config.search_budget,
            learning_rate=config.adam_learning_rate,
        ),
        "coordinate_cem": coordinate_cem_search(
            score,
            initial,
            bits=config.bits,
            population_size=config.population_size,
            generations=config.generations,
            elite_fraction=config.cem_elite_fraction,
            initial_std=config.cem_initial_std,
            minimum_std=config.cem_minimum_std,
            seed=seed + 101,
        ),
        "subset_mutation": subset_mutation_search(
            score,
            source,
            initial_indices,
            bits=config.bits,
            population_size=config.population_size,
            generations=config.generations,
            mutation_swaps=config.mutation_swaps,
            seed=seed + 202,
        ),
    }

    initial_histories = torch.stack(
        [results[method].best_loss_history[0] for method in METHODS]
    )
    paired_loss_delta = float(
        (initial_histories - initial_histories[:1]).abs().max().item()
    )
    budgets = {results[method].decoder_evaluations_per_cloud for method in METHODS}
    if paired_loss_delta > 1e-6 or len(budgets) != 1:
        raise RuntimeError("search comparison lost paired initialization or budget")

    methods: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        search_result = results[method]
        diagnostics = _diagnostics(
            search_result,
            target,
            score,
            config,
            seed=seed + 1_000 + method_index,
        )
        methods[method] = {
            "curve": _curve_json(search_result),
            "diagnostics": diagnostics,
            "mean_gain_from_initial": float(
                (initial_histories[method_index] - search_result.losses).mean().item()
            ),
            "improved_cloud_fraction": float(
                (search_result.losses < initial_histories[method_index])
                .float()
                .mean()
                .item()
            ),
            "decoder_evaluations_per_cloud": (
                search_result.decoder_evaluations_per_cloud
            ),
            "gradient_free": method != "adam_ste",
            "strict_subset": method == "subset_mutation",
        }

    per_cloud = []
    for cloud_index in range(len(points)):
        initial_loss = float(initial_histories[0, cloud_index].item())
        method_values = {}
        for method in METHODS:
            best_loss = float(results[method].losses[cloud_index].item())
            method_values[method] = {
                "best_chamfer": best_loss,
                "gain_from_paired_initialization": initial_loss - best_loss,
                "relative_gain": (initial_loss - best_loss) / max(initial_loss, 1e-12),
            }
        per_cloud.append(
            {
                "sample_id": cloud_index,
                "initial_chamfer": initial_loss,
                "methods": method_values,
            }
        )

    artifact = {
        f"{split}_{start_kind}_{method}_coordinates": results[method].coordinates.cpu()
        for method in METHODS
    }
    artifact[f"{split}_{start_kind}_initial_coordinates"] = initial.cpu()
    artifact[f"{split}_{start_kind}_initial_indices"] = initial_indices.cpu()
    assert results["subset_mutation"].indices is not None
    artifact[f"{split}_{start_kind}_subset_indices"] = results[
        "subset_mutation"
    ].indices.cpu()
    return (
        {
            "initial_constellation_hash": _tensor_hash(initial),
            "paired_initialization": True,
            "maximum_initial_loss_delta": paired_loss_delta,
            "initial_loss_match_tolerance": 1e-6,
            "matched_search_budget": True,
            "search_decoder_evaluations_per_cloud": budgets.pop(),
            "methods": methods,
            "per_cloud": per_cloud,
        },
        artifact,
    )


def run_gradient_free_experiment(
    config: GradientFreeExperimentConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train one decoder, freeze it, and run paired search comparisons."""

    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_points = _load_points(config.train_samples, config, "train", device)
    decoder = VariableConstellationDecoder(
        config.num_points,
        config.constellation_size,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)

    started = time.perf_counter()
    decoder_source = "trained"
    if config.decoder_checkpoint is None:
        decoder_history = _train_decoder(decoder, train_points, config)
        torch.save(
            {"model": decoder.state_dict(), "history": decoder_history},
            output_dir / "decoder.pt",
        )
    else:
        checkpoint_path = Path(config.decoder_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"decoder checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        decoder.load_state_dict(checkpoint.get("model", checkpoint))
        decoder_history = checkpoint.get("history", [])
        decoder_source = str(checkpoint_path)

    decoder.eval().requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.grad = None
    decoder_hash_before = _state_hash(decoder)
    evaluation: dict[str, Any] = {}
    artifacts: dict[str, Tensor] = {}
    for split_index, (split, sample_count) in enumerate(
        (
            ("validation", config.validation_samples),
            ("parameter_ood", config.parameter_ood_samples),
        )
    ):
        points = _load_points(sample_count, config, split, device)
        evaluation[split] = {}
        for start_index, start_kind in enumerate(config.start_kinds):
            start_result, start_artifacts = _evaluate_start(
                decoder,
                points,
                config,
                split=split,
                start_kind=start_kind,
                seed=config.seed + split_index * 10_000 + start_index * 1_000,
            )
            evaluation[split][start_kind] = start_result
            artifacts.update(start_artifacts)

    decoder_hash_after = _state_hash(decoder)
    decoder_unchanged = decoder_hash_before == decoder_hash_after
    if not decoder_unchanged:
        raise RuntimeError("frozen decoder changed during search")
    torch.save(artifacts, output_dir / "search_results.pt")

    result = {
        "experiment": "011_gradient_free_search",
        "config": asdict(config),
        "device": str(device),
        "fixed_operating_point": {
            "input_size": config.input_size,
            "constellation_size": config.constellation_size,
            "coordinate_payload_bits": 3 * config.constellation_size * config.bits,
        },
        "decoder_source": decoder_source,
        "decoder_history": decoder_history,
        "decoder_hash_before_search": decoder_hash_before,
        "decoder_hash_after_search": decoder_hash_after,
        "decoder_unchanged": decoder_unchanged,
        "decoder_trainable_parameter_count_during_search": sum(
            parameter.numel()
            for parameter in decoder.parameters()
            if parameter.requires_grad
        ),
        "score_contract": "decoder reconstruction Chamfer only",
        "message_contract": {
            "coordinate_only": True,
            "exact_final_quantization": True,
            "subset_is_unique_quantized_input_members": True,
        },
        "budget_definition": {
            "per_cloud_search_decoder_evaluations": config.search_budget,
            "diagnostic_evaluations_excluded": True,
            "gradient_evaluation_counts_as_one_decoder_evaluation": True,
        },
        "evaluation": evaluation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_011_gradient_free_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--decoder-checkpoint")
    args = parser.parse_args()

    config = GradientFreeExperimentConfig.from_json(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.decoder_checkpoint is not None:
        config = replace(config, decoder_checkpoint=args.decoder_checkpoint)
    result = run_gradient_free_experiment(config, device_name=args.device)
    print(
        json.dumps(
            {
                "device": result["device"],
                "decoder_unchanged": result["decoder_unchanged"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
