"""Train and evaluate decoder-population constellation cross-play (Experiment 010)."""

from __future__ import annotations

import argparse
import copy
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
from pointconstellation.models.crossplay import (
    DecoderPopulation,
    aggregate_decoder_losses,
    make_crossplay_refiner,
    make_decoder_population,
    reconstruction_loss_matrix,
)
from pointconstellation.models.refiner import CompetitiveConstellationRefiner
from pointconstellation.quantization import (
    quantization_step,
    quantize_coordinates,
)
from pointconstellation.train import select_device, set_seed

ENCODER_NAMES = ("matched_single_decoder", "population")


@dataclass(frozen=True)
class CrossplayExperimentConfig:
    num_points: int = 32
    input_sizes: tuple[int, ...] = (16, 32)
    constellation_sizes: tuple[int, ...] = (4, 8)
    bits: int = 10
    train_samples: int = 28
    validation_samples: int = 7
    parameter_ood_samples: int = 7
    batch_size: int = 7
    population_size: int = 3
    decoder_epochs: int = 1
    refiner_epochs: int = 1
    decoder_learning_rate: float = 1e-3
    refiner_learning_rate: float = 1e-3
    worst_decoder_weight: float = 0.25
    fps_probability: float = 0.5
    feature_width: int = 24
    num_heads: int = 4
    num_layers: int = 1
    recurrent_steps: int = 2
    responsibility_temperature: float = 0.2
    maximum_update: float = 0.1
    perturbation_bins: int = 1
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_010_crossplay_smoke"

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
                self.refiner_epochs,
            )
            < 1
        ):
            raise ValueError("sample counts, batch size, and epochs must be positive")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if min(self.decoder_learning_rate, self.refiner_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if self.worst_decoder_weight < 0:
            raise ValueError("worst_decoder_weight cannot be negative")
        if not 0 <= self.fps_probability <= 1:
            raise ValueError("fps_probability must be between zero and one")
        if self.feature_width < 4 or self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if self.num_layers < 1 or self.recurrent_steps < 1:
            raise ValueError("num_layers and recurrent_steps must be positive")
        if self.responsibility_temperature <= 0 or self.maximum_update <= 0:
            raise ValueError("temperature and maximum update must be positive")
        if self.perturbation_bins < 1:
            raise ValueError("perturbation_bins must be positive")

    @classmethod
    def from_json(cls, path: Path) -> CrossplayExperimentConfig:
        values = json.loads(path.read_text())
        for key in ("input_sizes", "constellation_sizes"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def _make_loaders(
    config: CrossplayExperimentConfig,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
]:
    datasets = {
        split: ProceduralPointCloudDataset(
            size,
            num_points=config.num_points,
            seed=config.seed,
            split=split,
        )
        for split, size in (
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


def _random_subset(
    points: Tensor,
    constellation_size: int,
    bits: int,
    generator: torch.Generator,
) -> Tensor:
    indices = torch.stack(
        [
            torch.randperm(points.shape[1], generator=generator)[:constellation_size]
            for _ in range(len(points))
        ]
    ).to(points.device)
    anchors = points.gather(1, indices[:, :, None].expand(-1, -1, 3))
    return quantize_coordinates(anchors, bits)


def _train_decoder_population(
    population: DecoderPopulation,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: CrossplayExperimentConfig,
) -> list[dict[str, Any]]:
    optimizers = [
        torch.optim.Adam(decoder.parameters(), lr=config.decoder_learning_rate)
        for decoder in population.decoders
    ]
    subset_generator = torch.Generator().manual_seed(config.seed + 101)
    history: list[dict[str, Any]] = []
    global_step = 0
    operating_points = tuple(
        (input_size, constellation_size)
        for input_size in config.input_sizes
        for constellation_size in config.constellation_sizes
    )
    for epoch in range(1, config.decoder_epochs + 1):
        population.train()
        totals = [0.0] * len(population)
        count = 0
        mode_counts = {"fps": 0, "random": 0}
        for batch in loader:
            target = batch["points"].to(device)
            input_size, constellation_size = operating_points[
                global_step % len(operating_points)
            ]
            source = target[:, :input_size]
            # Low-discrepancy scheduling makes even a short smoke run match the
            # requested FPS proportion instead of occasionally sampling one mode.
            use_fps = math.ceil((global_step + 1) * config.fps_probability) > math.ceil(
                global_step * config.fps_probability
            )
            if use_fps:
                constellation = _fps(source, constellation_size, config.bits)
                mode_counts["fps"] += len(target)
            else:
                constellation = _random_subset(
                    source, constellation_size, config.bits, subset_generator
                )
                mode_counts["random"] += len(target)

            # Every independently initialized member receives the exact same legal
            # coordinate message, target, and data order.
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            losses = [
                chamfer_squared(
                    decoder(constellation, num_output_points=config.num_points),
                    target,
                )
                for decoder in population.decoders
            ]
            for loss, optimizer in zip(losses, optimizers, strict=True):
                loss.backward()
                optimizer.step()
            for index, loss in enumerate(losses):
                totals[index] += float(loss.item()) * len(target)
            count += len(target)
            global_step += 1

        record: dict[str, Any] = {
            "epoch": epoch,
            "decoder_chamfer_rmse": [math.sqrt(total / count) for total in totals],
            "message_mode_counts": mode_counts,
        }
        history.append(record)
        print(json.dumps({"stage": "decoder_population", **record}))
    return history


def _train_refiners(
    matched: CompetitiveConstellationRefiner,
    population_trained: CompetitiveConstellationRefiner,
    population: DecoderPopulation,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: CrossplayExperimentConfig,
) -> list[dict[str, Any]]:
    population.eval().requires_grad_(False)
    matched_optimizer = torch.optim.Adam(
        matched.parameters(), lr=config.refiner_learning_rate
    )
    population_optimizer = torch.optim.Adam(
        population_trained.parameters(), lr=config.refiner_learning_rate
    )
    history: list[dict[str, Any]] = []
    global_step = 0
    operating_points = tuple(
        (input_size, constellation_size)
        for input_size in config.input_sizes
        for constellation_size in config.constellation_sizes
    )
    for epoch in range(1, config.refiner_epochs + 1):
        matched.train()
        population_trained.train()
        matched_total = 0.0
        population_total = 0.0
        count = 0
        for batch in loader:
            target = batch["points"].to(device)
            input_size, constellation_size = operating_points[
                global_step % len(operating_points)
            ]
            source = target[:, :input_size]

            matched_optimizer.zero_grad(set_to_none=True)
            matched_message = matched(source, constellation_size)
            matched_loss = chamfer_squared(
                population.decoders[0](
                    matched_message, num_output_points=config.num_points
                ),
                target,
            )
            matched_loss.backward()
            matched_optimizer.step()

            population_optimizer.zero_grad(set_to_none=True)
            population_message = population_trained(source, constellation_size)
            decoder_losses = torch.stack(
                [
                    chamfer_squared(reconstruction, target)
                    for reconstruction in population.forward_all(
                        population_message, num_output_points=config.num_points
                    )
                ]
            )
            population_loss = aggregate_decoder_losses(
                decoder_losses, worst_weight=config.worst_decoder_weight
            )
            population_loss.backward()
            population_optimizer.step()

            matched_total += float(matched_loss.item()) * len(target)
            population_total += float(population_loss.item()) * len(target)
            count += len(target)
            global_step += 1

        record = {
            "epoch": epoch,
            "matched_objective_rmse": math.sqrt(matched_total / count),
            "population_objective_rmse": math.sqrt(population_total / count),
        }
        history.append(record)
        print(json.dumps({"stage": "crossplay_refiners", **record}))
    return history


def _resampled_sources(
    target: Tensor,
    input_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    first_indices = []
    second_indices = []
    for _ in range(len(target)):
        first_indices.append(
            torch.randperm(target.shape[1], generator=generator)[:input_size]
        )
        second_indices.append(
            torch.randperm(target.shape[1], generator=generator)[:input_size]
        )
    first = torch.stack(first_indices).to(target.device)
    second = torch.stack(second_indices).to(target.device)
    return (
        target.gather(1, first[:, :, None].expand(-1, -1, 3)),
        target.gather(1, second[:, :, None].expand(-1, -1, 3)),
    )


def _message_set_squared(first: Tensor, second: Tensor) -> Tensor:
    distances = pairwise_squared(first, second)
    return 0.5 * (distances.amin(2).mean(1) + distances.amin(1).mean(1))


def _perturb_message(
    message: Tensor,
    config: CrossplayExperimentConfig,
    generator: torch.Generator,
) -> Tensor:
    directions = torch.randint(
        -1,
        2,
        message.shape,
        generator=generator,
        dtype=torch.int64,
    ).to(device=message.device, dtype=message.dtype)
    # A vanishing perturbation would understate sensitivity for tiny messages.
    directions[:, 0, 0] = torch.where(
        directions[:, 0, 0] == 0,
        torch.ones_like(directions[:, 0, 0]),
        directions[:, 0, 0],
    )
    delta = config.perturbation_bins * quantization_step(config.bits) * directions
    return quantize_coordinates(message + delta, config.bits)


def _matrix_rmse(totals: Tensor, count: int) -> list[list[float]]:
    return (totals / count).sqrt().tolist()


def _evaluate_crossplay(
    refiners: tuple[CompetitiveConstellationRefiner, ...],
    population: DecoderPopulation,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: CrossplayExperimentConfig,
    *,
    split_seed: int,
    input_size: int,
    constellation_size: int,
) -> dict[str, Any]:
    for refiner in refiners:
        refiner.eval()
    population.eval()
    matrix_totals = torch.zeros((len(refiners), len(population)), dtype=torch.float64)
    perturbed_totals = torch.zeros_like(matrix_totals)
    stability_totals = torch.zeros(len(refiners), dtype=torch.float64)
    maximum_lattice_error = torch.zeros(len(refiners), dtype=torch.float64)
    perturbation_generator = torch.Generator().manual_seed(split_seed)
    resampling_generator = torch.Generator().manual_seed(split_seed + 1)
    count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["points"].to(device)
            source = target[:, :input_size]
            messages = tuple(
                refiner(source, constellation_size) for refiner in refiners
            )
            batch_matrix = reconstruction_loss_matrix(
                messages,
                population,
                target,
                num_output_points=config.num_points,
            )
            matrix_totals += batch_matrix.detach().cpu().to(torch.float64) * len(target)

            # Apply identical lattice directions to both messages so the
            # sensitivity comparison is paired rather than noise-confounded.
            perturbation_state = perturbation_generator.get_state()
            perturbed_messages_list = []
            state_after_one = None
            for message in messages:
                perturbation_generator.set_state(perturbation_state)
                perturbed_messages_list.append(
                    _perturb_message(message, config, perturbation_generator)
                )
                if state_after_one is None:
                    state_after_one = perturbation_generator.get_state()
            assert state_after_one is not None
            perturbation_generator.set_state(state_after_one)
            perturbed_messages = tuple(perturbed_messages_list)
            perturbed_matrix = reconstruction_loss_matrix(
                perturbed_messages,
                population,
                target,
                num_output_points=config.num_points,
            )
            perturbed_totals += perturbed_matrix.detach().cpu().to(torch.float64) * len(
                target
            )

            source_a, source_b = _resampled_sources(
                target, input_size, resampling_generator
            )
            for index, refiner in enumerate(refiners):
                message_a = refiner(source_a, constellation_size)
                message_b = refiner(source_b, constellation_size)
                stability_totals[index] += float(
                    _message_set_squared(message_a, message_b).sum().item()
                )

            step = quantization_step(config.bits)
            for index, message in enumerate(messages):
                lattice = (message + 1.0) / step
                error = (lattice - lattice.round()).abs().amax().double().cpu()
                maximum_lattice_error[index] = torch.maximum(
                    maximum_lattice_error[index], error
                )
            count += len(target)

    matrix = _matrix_rmse(matrix_totals, count)
    perturbed = _matrix_rmse(perturbed_totals, count)
    mean_cross_decoder = {
        name: math.sqrt(float(matrix_totals[index].mean().item()) / count)
        for index, name in enumerate(ENCODER_NAMES)
    }
    worst_decoder = {}
    for index, name in enumerate(ENCODER_NAMES):
        decoder_index = int(matrix_totals[index].argmax().item())
        worst_decoder[name] = {
            "decoder_index": decoder_index,
            "chamfer_rmse": matrix[index][decoder_index],
        }
    delta_matrix = [
        [
            perturbed[row][column] - matrix[row][column]
            for column in range(len(population))
        ]
        for row in range(len(refiners))
    ]
    population_delta_by_decoder = [
        100.0 * (matrix[1][column] - matrix[0][column]) / max(matrix[0][column], 1e-12)
        for column in range(len(population))
    ]
    return {
        "input_size": input_size,
        "constellation_size": constellation_size,
        "coordinate_payload_bits": 3 * constellation_size * config.bits,
        "encoder_rows": list(ENCODER_NAMES),
        "decoder_columns": [f"decoder_{index}" for index in range(len(population))],
        "crossplay_chamfer_rmse_matrix": matrix,
        "paired_performance": {
            "matched_encoder_decoder_0_chamfer_rmse": matrix[0][0],
            "population_encoder_decoder_0_chamfer_rmse": matrix[1][0],
            "population_encoder_partner_mean_chamfer_rmse": mean_cross_decoder[
                "population"
            ],
        },
        "mean_cross_decoder_chamfer_rmse": mean_cross_decoder,
        "population_delta_vs_matched_by_decoder_percent": (population_delta_by_decoder),
        "population_mean_normalized_delta_percent": sum(population_delta_by_decoder)
        / len(population_delta_by_decoder),
        "worst_decoder": worst_decoder,
        "perturbation_sensitivity": {
            "bins": config.perturbation_bins,
            "perturbed_chamfer_rmse_matrix": perturbed,
            "delta_chamfer_rmse_matrix": delta_matrix,
        },
        "message_resampling_stability_rmse": {
            name: math.sqrt(float(stability_totals[index].item()) / count)
            for index, name in enumerate(ENCODER_NAMES)
        },
        "maximum_quantization_lattice_error": {
            name: float(maximum_lattice_error[index].item())
            for index, name in enumerate(ENCODER_NAMES)
        },
    }


def run_crossplay_experiment(
    config: CrossplayExperimentConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the matched single-decoder versus decoder-population experiment."""

    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, ood_loader = _make_loaders(config)
    population = make_decoder_population(
        config.population_size,
        max_output_points=config.num_points,
        max_constellation_size=max(config.constellation_sizes),
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        seed=config.seed + 1000,
    ).to(device)

    started = time.perf_counter()
    independent_initial_hashes = [
        _state_hash(decoder) for decoder in population.decoders
    ]
    if len(set(independent_initial_hashes)) != len(independent_initial_hashes):
        raise RuntimeError("decoder population did not initialize independently")
    decoder_history = _train_decoder_population(
        population, train_loader, device, config
    )
    decoder_hashes_before_refiners = [
        _state_hash(decoder) for decoder in population.decoders
    ]

    set_seed(config.seed + 2000)
    base_refiner = make_crossplay_refiner(
        max(config.constellation_sizes),
        bits=config.bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        recurrent_steps=config.recurrent_steps,
        responsibility_temperature=config.responsibility_temperature,
        maximum_update=config.maximum_update,
    ).to(device)
    matched_refiner = copy.deepcopy(base_refiner)
    population_refiner = copy.deepcopy(base_refiner)
    del base_refiner
    initial_refiner_hash = _state_hash(matched_refiner)
    refiner_initialization_matched = initial_refiner_hash == _state_hash(
        population_refiner
    )
    if not refiner_initialization_matched:
        raise RuntimeError("cross-play refiner initializations do not match")

    refiner_history = _train_refiners(
        matched_refiner,
        population_refiner,
        population,
        train_loader,
        device,
        config,
    )
    decoder_hashes_after_refiners = [
        _state_hash(decoder) for decoder in population.decoders
    ]
    decoders_unchanged = decoder_hashes_before_refiners == decoder_hashes_after_refiners
    if not decoders_unchanged:
        raise RuntimeError("a frozen decoder changed during refiner training")

    refiners = (matched_refiner, population_refiner)
    evaluation: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "parameter_ood": [],
    }
    for input_size in config.input_sizes:
        for constellation_size in config.constellation_sizes:
            for split_index, (split, loader) in enumerate(
                (
                    ("validation", validation_loader),
                    ("parameter_ood", ood_loader),
                )
            ):
                evaluation[split].append(
                    _evaluate_crossplay(
                        refiners,
                        population,
                        loader,
                        device,
                        config,
                        split_seed=(
                            config.seed
                            + 10_000
                            + split_index * 1000
                            + input_size * 10
                            + constellation_size
                        ),
                        input_size=input_size,
                        constellation_size=constellation_size,
                    )
                )

    result = {
        "config": asdict(config),
        "device": str(device),
        "message_contract": {
            "payload": "Kx3 coordinates",
            "bits_per_coordinate": config.bits,
            "exactly_quantized": True,
            "hidden_features_transmitted": False,
        },
        "encoder_rows": list(ENCODER_NAMES),
        "decoder_initial_hashes": independent_initial_hashes,
        "decoder_hashes_before_refiners": decoder_hashes_before_refiners,
        "decoder_hashes_after_refiners": decoder_hashes_after_refiners,
        "decoder_population_initialized_independently": len(
            set(independent_initial_hashes)
        )
        == len(independent_initial_hashes),
        "decoders_unchanged_during_refiner_training": decoders_unchanged,
        "refiner_initialization_hash": initial_refiner_hash,
        "refiner_initialization_matched": refiner_initialization_matched,
        "decoder_parameter_count_each": [
            sum(parameter.numel() for parameter in decoder.parameters())
            for decoder in population.decoders
        ],
        "refiner_parameter_count_each": sum(
            parameter.numel() for parameter in matched_refiner.parameters()
        ),
        "decoder_history": decoder_history,
        "refiner_history": refiner_history,
        "evaluation": evaluation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    for index, decoder in enumerate(population.decoders):
        torch.save(
            {"model": decoder.state_dict(), "config": asdict(config)},
            output_dir / f"decoder_{index}.pt",
        )
    torch.save(
        {"model": matched_refiner.state_dict(), "config": asdict(config)},
        output_dir / "matched_refiner.pt",
    )
    torch.save(
        {"model": population_refiner.state_dict(), "config": asdict(config)},
        output_dir / "population_refiner.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_010_crossplay_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = CrossplayExperimentConfig.from_json(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    result = run_crossplay_experiment(config, device_name=args.device)
    print(
        json.dumps(
            {
                "device": result["device"],
                "decoders_unchanged": result[
                    "decoders_unchanged_during_refiner_training"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
