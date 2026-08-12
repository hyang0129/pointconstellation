"""Compare mass-aware objectives under a matched frozen-decoder protocol."""

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
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.refiner import CompetitiveConstellationRefiner
from pointconstellation.models.transport import (
    BalancedResponsibilityRefiner,
    balanced_anchor_transport_loss,
    balanced_transport_squared,
    density_aware_chamfer_squared,
    soft_anchor_responsibilities,
)
from pointconstellation.quantization import quantization_step, quantize_ste
from pointconstellation.train import select_device, set_seed

OBJECTIVES = ("chamfer", "density_aware", "balanced_transport")


@dataclass(frozen=True)
class TransportExperimentConfig:
    """Configuration for the fixed-N/K objective comparison."""

    num_points: int = 24
    input_size: int = 24
    constellation_size: int = 4
    bits: int = 8
    train_samples: int = 12
    validation_samples: int = 4
    parameter_ood_samples: int = 4
    batch_size: int = 4
    decoder_epochs: int = 1
    refiner_epochs: int = 1
    decoder_learning_rate: float = 1e-3
    refiner_learning_rate: float = 1e-3
    feature_width: int = 16
    num_heads: int = 4
    num_layers: int = 1
    recurrent_steps: int = 2
    responsibility_temperature: float = 0.2
    maximum_update: float = 0.1
    use_decoder_gradient: bool = False
    objectives: tuple[str, ...] = OBJECTIVES
    sinkhorn_epsilon: float = 0.08
    sinkhorn_iterations: int = 30
    density_temperature: float = 0.05
    anchor_transport_weight: float = 0.1
    mass_temperature: float = 0.05
    decoder_checkpoint: str | None = None
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_008_transport_smoke"

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
                self.refiner_epochs,
                self.num_layers,
                self.recurrent_steps,
                self.sinkhorn_iterations,
            )
            < 1
        ):
            raise ValueError(
                "sample counts, epochs, layers, and steps must be positive"
            )
        if self.feature_width < 4 or self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if min(self.decoder_learning_rate, self.refiner_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if (
            min(
                self.responsibility_temperature,
                self.maximum_update,
                self.sinkhorn_epsilon,
                self.density_temperature,
                self.mass_temperature,
            )
            <= 0
        ):
            raise ValueError(
                "temperatures, epsilon, and maximum update must be positive"
            )
        if self.anchor_transport_weight < 0:
            raise ValueError("anchor_transport_weight cannot be negative")
        if not self.objectives or len(set(self.objectives)) != len(self.objectives):
            raise ValueError("objectives must be nonempty and unique")
        unknown = set(self.objectives) - set(OBJECTIVES)
        if unknown:
            raise ValueError(f"unknown objectives: {sorted(unknown)}")

    @classmethod
    def from_json(cls, path: Path) -> TransportExperimentConfig:
        values = json.loads(path.read_text())
        if "objectives" in values:
            values["objectives"] = tuple(values["objectives"])
        return cls(**values)


def _make_loader(
    config: TransportExperimentConfig,
    split: str,
    sample_count: int,
    *,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    dataset = ProceduralPointCloudDataset(
        sample_count,
        num_points=config.num_points,
        seed=config.seed,
        split=split,
    )
    generator = torch.Generator().manual_seed(config.seed + 8_008)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


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
    return quantize_ste(anchors, bits, training=False, jitter=False)


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _objective_loss(
    objective: str,
    reconstruction: Tensor,
    target: Tensor,
    constellation: Tensor,
    config: TransportExperimentConfig,
) -> Tensor:
    if objective == "chamfer":
        return chamfer_squared(reconstruction, target)
    if objective == "density_aware":
        return density_aware_chamfer_squared(
            reconstruction,
            target,
            temperature=config.density_temperature,
        )
    if objective == "balanced_transport":
        reconstruction_loss = balanced_transport_squared(
            reconstruction,
            target,
            epsilon=config.sinkhorn_epsilon,
            iterations=config.sinkhorn_iterations,
        )
        anchor_loss = balanced_anchor_transport_loss(
            constellation,
            target,
            epsilon=config.sinkhorn_epsilon,
            iterations=config.sinkhorn_iterations,
        )
        return reconstruction_loss + config.anchor_transport_weight * anchor_loss
    raise ValueError(f"unknown objective: {objective}")


def _train_decoder(
    decoder: VariableConstellationDecoder,
    config: TransportExperimentConfig,
    device: torch.device,
) -> list[dict[str, float | int]]:
    loader = _make_loader(config, "train", config.train_samples, shuffle=True)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.decoder_learning_rate)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.decoder_epochs + 1):
        decoder.train()
        total = 0.0
        count = 0
        for batch in loader:
            target = batch["points"].to(device)
            source = target[:, : config.input_size]
            constellation = _fps(source, config.constellation_size, config.bits)
            constellation = quantize_ste(
                constellation, config.bits, training=True, jitter=True
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


def _train_refiner_arm(
    objective: str,
    refiner: CompetitiveConstellationRefiner,
    decoder: VariableConstellationDecoder,
    config: TransportExperimentConfig,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], str]:
    decoder.eval().requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.grad = None
    loader = _make_loader(config, "train", config.train_samples, shuffle=True)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=config.refiner_learning_rate)
    order_digest = hashlib.sha256()
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.refiner_epochs + 1):
        refiner.train()
        total = 0.0
        count = 0
        for batch in loader:
            sample_ids = batch["sample_id"]
            order_digest.update(sample_ids.detach().cpu().numpy().tobytes())
            target = batch["points"].to(device)
            source = target[:, : config.input_size]
            optimizer.zero_grad(set_to_none=True)
            _, states = refiner(
                source,
                config.constellation_size,
                decoder=decoder,
                target=source,
                num_output_points=config.num_points,
                return_history=True,
            )
            losses = []
            for constellation in states[1:]:
                reconstruction = decoder(
                    constellation, num_output_points=config.num_points
                )
                losses.append(
                    _objective_loss(
                        objective,
                        reconstruction,
                        target,
                        constellation,
                        config,
                    )
                )
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(target)
            count += len(target)
        record = {
            "epoch": epoch,
            "training_objective": total / count,
        }
        history.append(record)
        print(json.dumps({"stage": "refiner", "objective": objective, **record}))
    return history, order_digest.hexdigest()


def _sample_metrics(
    reconstruction: Tensor,
    target: Tensor,
    constellation: Tensor,
    config: TransportExperimentConfig,
) -> dict[str, Tensor]:
    reconstruction_distances = pairwise_squared(reconstruction, target)
    chamfer = 0.5 * (
        reconstruction_distances.amin(2).mean(1)
        + reconstruction_distances.amin(1).mean(1)
    )
    anchor_distances = pairwise_squared(constellation, target)
    coverage = anchor_distances.amin(1).mean(1)
    surface = anchor_distances.amin(2).mean(1)
    responsibilities = soft_anchor_responsibilities(
        constellation, target, temperature=config.mass_temperature
    )
    mass_fraction = responsibilities.mean(dim=2)
    expected = 1.0 / constellation.shape[1]
    mass_imbalance = ((mass_fraction - expected) / expected).square().mean(dim=1).sqrt()
    lattice_step = quantization_step(config.bits)
    lattice_error = (
        ((constellation + 1.0) / lattice_step)
        .sub(((constellation + 1.0) / lattice_step).round())
        .abs()
        .amax(dim=(1, 2))
    )
    return {
        "chamfer": chamfer,
        "coverage": coverage,
        "surface": surface,
        "anchor_mass_imbalance": mass_imbalance,
        "maximum_lattice_error": lattice_error,
    }


def _evaluate_curve(
    refiner: CompetitiveConstellationRefiner,
    decoder: VariableConstellationDecoder,
    config: TransportExperimentConfig,
    device: torch.device,
    split: str,
    sample_count: int,
) -> list[dict[str, float | int]]:
    loader = _make_loader(config, split, sample_count, shuffle=False)
    refiner.eval()
    decoder.eval()
    names = (
        "chamfer",
        "coverage",
        "surface",
        "anchor_mass_imbalance",
        "maximum_lattice_error",
    )
    totals = [{name: 0.0 for name in names} for _ in range(config.recurrent_steps + 1)]
    count = 0
    for batch in loader:
        target = batch["points"].to(device)
        source = target[:, : config.input_size]
        with torch.no_grad():
            _, states = refiner(
                source,
                config.constellation_size,
                decoder=decoder,
                target=source,
                num_output_points=config.num_points,
                return_history=True,
            )
            for step, constellation in enumerate(states):
                reconstruction = decoder(
                    constellation, num_output_points=config.num_points
                )
                metrics = _sample_metrics(reconstruction, target, constellation, config)
                for name, values in metrics.items():
                    if name == "maximum_lattice_error":
                        totals[step][name] = max(
                            totals[step][name], float(values.max().item())
                        )
                    else:
                        totals[step][name] += float(values.sum().item())
        count += len(target)

    curve: list[dict[str, float | int]] = []
    for step, values in enumerate(totals):
        chamfer = values["chamfer"] / count
        coverage = values["coverage"] / count
        surface = values["surface"] / count
        curve.append(
            {
                "step": step,
                "reconstruction_chamfer": chamfer,
                "reconstruction_chamfer_rmse": math.sqrt(chamfer),
                "coverage": coverage,
                "coverage_rmse": math.sqrt(coverage),
                "surface": surface,
                "surface_rmse": math.sqrt(surface),
                "anchor_mass_imbalance": values["anchor_mass_imbalance"] / count,
                "maximum_lattice_error": values["maximum_lattice_error"],
            }
        )
    return curve


def run_transport_experiment(
    config: TransportExperimentConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run all objective arms with one frozen decoder and matched refiners."""

    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        decoder_history = _train_decoder(decoder, config, device)
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

    base_refiner = CompetitiveConstellationRefiner(
        config.constellation_size,
        bits=config.bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        recurrent_steps=config.recurrent_steps,
        responsibility_temperature=config.responsibility_temperature,
        maximum_update=config.maximum_update,
        use_decoder_gradient=config.use_decoder_gradient,
    ).to(device)
    base_state = {
        name: value.detach().clone()
        for name, value in base_refiner.state_dict().items()
    }

    arms: dict[str, Any] = {}
    initial_hashes: dict[str, str] = {}
    order_hashes: dict[str, str] = {}
    decoder_hashes_after_arms: dict[str, str] = {}
    for objective in config.objectives:
        set_seed(config.seed + 8_008)
        refiner_type = (
            BalancedResponsibilityRefiner
            if objective == "balanced_transport"
            else CompetitiveConstellationRefiner
        )
        refiner_kwargs: dict[str, Any] = {}
        if objective == "balanced_transport":
            refiner_kwargs = {
                "sinkhorn_epsilon": config.sinkhorn_epsilon,
                "sinkhorn_iterations": config.sinkhorn_iterations,
            }
        refiner = refiner_type(
            config.constellation_size,
            bits=config.bits,
            feature_width=config.feature_width,
            num_heads=config.num_heads,
            recurrent_steps=config.recurrent_steps,
            responsibility_temperature=config.responsibility_temperature,
            maximum_update=config.maximum_update,
            use_decoder_gradient=config.use_decoder_gradient,
            **refiner_kwargs,
        ).to(device)
        refiner.load_state_dict(base_state)
        initial_hashes[objective] = _state_hash(refiner)
        history, order_hash = _train_refiner_arm(
            objective, refiner, decoder, config, device
        )
        decoder_hashes_after_arms[objective] = _state_hash(decoder)
        if decoder_hashes_after_arms[objective] != decoder_hash_before:
            raise RuntimeError(f"frozen decoder changed during {objective} training")
        order_hashes[objective] = order_hash
        arms[objective] = {
            "history": history,
            "validation_curve": _evaluate_curve(
                refiner,
                decoder,
                config,
                device,
                "validation",
                config.validation_samples,
            ),
            "parameter_ood_curve": _evaluate_curve(
                refiner,
                decoder,
                config,
                device,
                "parameter_ood",
                config.parameter_ood_samples,
            ),
            "final_refiner_hash": _state_hash(refiner),
        }
        torch.save(
            {
                "model": refiner.state_dict(),
                "objective": objective,
                "config": asdict(config),
            },
            output_dir / f"refiner_{objective}.pt",
        )

    decoder_hash_after = _state_hash(decoder)
    decoder_unchanged = decoder_hash_before == decoder_hash_after
    if not decoder_unchanged:
        raise RuntimeError("frozen decoder changed during objective comparison")
    matched_initialization = len(set(initial_hashes.values())) == 1
    matched_data_order = len(set(order_hashes.values())) == 1
    if not matched_initialization or not matched_data_order:
        raise RuntimeError("objective arms did not receive a matched comparison")

    result = {
        "experiment": "008_mass_aware_transport_objectives",
        "config": asdict(config),
        "device": str(device),
        "fixed_operating_point": {
            "input_size": config.input_size,
            "constellation_size": config.constellation_size,
            "coordinate_payload_bits": 3 * config.constellation_size * config.bits,
        },
        "decoder_source": decoder_source,
        "decoder_history": decoder_history,
        "decoder_hash_before_refiners": decoder_hash_before,
        "decoder_hash_after_refiners": decoder_hash_after,
        "decoder_hashes_after_arms": decoder_hashes_after_arms,
        "decoder_unchanged": decoder_unchanged,
        "decoder_trainable_parameter_count_after_freeze": sum(
            parameter.numel()
            for parameter in decoder.parameters()
            if parameter.requires_grad
        ),
        "refiner_initial_hashes": initial_hashes,
        "training_order_hashes": order_hashes,
        "matched_initialization": matched_initialization,
        "matched_data_order": matched_data_order,
        "message_contract": {
            "coordinate_only": True,
            "coordinate_shape": [config.constellation_size, 3],
            "exact_final_quantization": True,
            "bits_per_coordinate": config.bits,
        },
        "diagnostic_definition": {
            "anchor_mass_imbalance": (
                "relative RMS deviation of unconstrained soft anchor mass from 1/K"
            ),
            "balanced_transport_anchor_weight": config.anchor_transport_weight,
        },
        "arms": arms,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_008_transport_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--decoder-checkpoint")
    args = parser.parse_args()

    config = TransportExperimentConfig.from_json(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.decoder_checkpoint is not None:
        config = replace(config, decoder_checkpoint=args.decoder_checkpoint)
    result = run_transport_experiment(config, device_name=args.device)
    print(
        json.dumps(
            {
                "device": result["device"],
                "decoder_unchanged": result["decoder_unchanged"],
                "matched_initialization": result["matched_initialization"],
                "matched_data_order": result["matched_data_order"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
