"""Train and evaluate the competitive semi-amortized constellation refiner."""

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
from pointconstellation.quantization import quantize_ste
from pointconstellation.train import select_device, set_seed


@dataclass(frozen=True)
class RefinerExperimentConfig:
    num_points: int = 64
    input_sizes: tuple[int, ...] = (32, 64)
    constellation_sizes: tuple[int, ...] = (4, 8)
    bits: int = 10
    train_samples: int = 56
    validation_samples: int = 14
    parameter_ood_samples: int = 14
    batch_size: int = 7
    decoder_epochs: int = 2
    refiner_epochs: int = 2
    decoder_learning_rate: float = 1e-3
    refiner_learning_rate: float = 1e-3
    feature_width: int = 48
    num_heads: int = 4
    num_layers: int = 1
    recurrent_steps: int = 3
    responsibility_temperature: float = 0.2
    maximum_update: float = 0.1
    use_decoder_gradient: bool = True
    decoder_checkpoint: str | None = None
    data_seed: int | None = None
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_005_refiner_smoke"

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
        if self.feature_width < 4 or self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if self.num_layers < 1 or self.recurrent_steps < 1:
            raise ValueError("num_layers and recurrent_steps must be positive")
        if min(self.decoder_learning_rate, self.refiner_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if self.responsibility_temperature <= 0 or self.maximum_update <= 0:
            raise ValueError("temperature and maximum update must be positive")

    @classmethod
    def from_json(cls, path: Path) -> RefinerExperimentConfig:
        values = json.loads(path.read_text())
        for key in ("input_sizes", "constellation_sizes"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def _make_loaders(
    config: RefinerExperimentConfig,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
]:
    data_seed = config.seed if config.data_seed is None else config.data_seed
    datasets = {
        split: ProceduralPointCloudDataset(
            size,
            num_points=config.num_points,
            seed=data_seed,
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


def _train_decoder(
    decoder: VariableConstellationDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: RefinerExperimentConfig,
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
            input_size = config.input_sizes[global_step % len(config.input_sizes)]
            constellation_size = config.constellation_sizes[
                global_step % len(config.constellation_sizes)
            ]
            source = target[:, :input_size]
            constellation = _fps(source, constellation_size, config.bits)
            constellation = quantize_ste(constellation, config.bits, training=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = decoder(constellation, num_output_points=config.num_points)
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


def _train_refiner(
    refiner: CompetitiveConstellationRefiner,
    decoder: VariableConstellationDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: RefinerExperimentConfig,
) -> tuple[list[dict[str, float | int]], str]:
    decoder.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=config.refiner_learning_rate)
    history: list[dict[str, float | int]] = []
    order_digest = hashlib.sha256()
    global_step = 0
    for epoch in range(1, config.refiner_epochs + 1):
        refiner.train()
        total = 0.0
        count = 0
        for batch in loader:
            order_digest.update(batch["sample_id"].cpu().numpy().tobytes())
            target = batch["points"].to(device)
            input_size = config.input_sizes[global_step % len(config.input_sizes)]
            constellation_size = config.constellation_sizes[
                global_step % len(config.constellation_sizes)
            ]
            source = target[:, :input_size]
            optimizer.zero_grad(set_to_none=True)
            _, states = refiner(
                source,
                constellation_size,
                decoder=decoder,
                # Decoder-gradient feedback is part of the encoder.  It may
                # only inspect points available at encoding time; the full
                # target remains valid for the outer training loss below.
                target=source,
                num_output_points=config.num_points,
                return_history=True,
            )
            step_losses = [
                chamfer_squared(
                    decoder(state, num_output_points=config.num_points), target
                )
                for state in states[1:]
            ]
            loss = torch.stack(step_losses).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(target)
            count += len(target)
            global_step += 1
        record = {
            "epoch": epoch,
            "training_refined_chamfer_rmse": math.sqrt(total / count),
        }
        history.append(record)
        print(json.dumps({"stage": "refiner", **record}))
    return history, order_digest.hexdigest()


def _sample_metrics(
    reconstruction: Tensor, target: Tensor, constellation: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    reconstruction_distances = pairwise_squared(reconstruction, target)
    chamfer = 0.5 * (
        reconstruction_distances.amin(2).mean(1)
        + reconstruction_distances.amin(1).mean(1)
    )
    anchor_distances = pairwise_squared(constellation, target)
    surface = anchor_distances.amin(2).mean(1)
    coverage = anchor_distances.amin(1).mean(1)
    return chamfer, surface, coverage


def _evaluate_curve(
    refiner: CompetitiveConstellationRefiner,
    decoder: VariableConstellationDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: RefinerExperimentConfig,
    *,
    input_size: int,
    constellation_size: int,
) -> list[dict[str, Any]]:
    refiner.eval()
    decoder.eval()
    totals = [
        {
            mode: {"chamfer": 0.0, "surface": 0.0, "coverage": 0.0}
            for mode in ("free", "strict_subset")
        }
        for _ in range(config.recurrent_steps + 1)
    ]
    count = 0
    for batch in loader:
        target = batch["points"].to(device)
        source = target[:, :input_size]
        with torch.no_grad():
            _, states = refiner(
                source,
                constellation_size,
                decoder=decoder,
                target=source,
                num_output_points=config.num_points,
                return_history=True,
            )
            for step, free in enumerate(states):
                constellations = {
                    "free": free,
                    "strict_subset": refiner.project_unique_to_input(free, source),
                }
                for mode, constellation in constellations.items():
                    reconstruction = decoder(
                        constellation, num_output_points=config.num_points
                    )
                    chamfer, surface, coverage = _sample_metrics(
                        reconstruction, target, constellation
                    )
                    totals[step][mode]["chamfer"] += float(chamfer.sum().item())
                    totals[step][mode]["surface"] += float(surface.sum().item())
                    totals[step][mode]["coverage"] += float(coverage.sum().item())
        count += len(target)

    curve = []
    for step, step_totals in enumerate(totals):
        record: dict[str, Any] = {"step": step}
        for mode, mode_totals in step_totals.items():
            means = {name: value / count for name, value in mode_totals.items()}
            record[mode] = {
                **means,
                "chamfer_rmse": math.sqrt(means["chamfer"]),
                "surface_rmse": math.sqrt(means["surface"]),
                "coverage_rmse": math.sqrt(means["coverage"]),
            }
        curve.append(record)
    return curve


def run_refiner_experiment(
    config: RefinerExperimentConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train/load the decoder, freeze it, train the refiner, and emit curves."""

    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder_train_loader, validation_loader, ood_loader = _make_loaders(config)
    decoder = VariableConstellationDecoder(
        config.num_points,
        max(config.constellation_sizes),
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)
    refiner = CompetitiveConstellationRefiner(
        max(config.constellation_sizes),
        bits=config.bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        recurrent_steps=config.recurrent_steps,
        responsibility_temperature=config.responsibility_temperature,
        maximum_update=config.maximum_update,
        use_decoder_gradient=config.use_decoder_gradient,
    ).to(device)

    started = time.perf_counter()
    decoder_source = "trained"
    if config.decoder_checkpoint is not None:
        checkpoint_path = Path(config.decoder_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"decoder checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        decoder.load_state_dict(checkpoint.get("model", checkpoint))
        decoder_history = checkpoint.get("history", [])
        decoder_source = str(checkpoint_path)
    else:
        decoder_history = _train_decoder(decoder, decoder_train_loader, device, config)
        torch.save(
            {"model": decoder.state_dict(), "history": decoder_history},
            output_dir / "decoder.pt",
        )

    decoder.eval().requires_grad_(False)
    decoder_hash_before = _state_hash(decoder)
    refiner_hash_before = _state_hash(refiner)
    # Decoder training consumes random numbers (including quantizer jitter).
    # Reset here so a freshly trained decoder arm and a checkpoint-loaded
    # control see the same refiner-time random stream.
    set_seed(config.seed)
    # Recreate the loader so decoder training cannot alter the refiner's batch
    # order.  Checkpoint-loaded and freshly trained decoder arms are therefore
    # matched under the same refiner seed.
    refiner_train_loader, _, _ = _make_loaders(config)
    refiner_history, refiner_training_order_hash = _train_refiner(
        refiner, decoder, refiner_train_loader, device, config
    )
    decoder_hash_after = _state_hash(decoder)
    decoder_unchanged = decoder_hash_before == decoder_hash_after
    if not decoder_unchanged:
        raise RuntimeError("frozen decoder changed during refiner training")

    evaluation: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "parameter_ood": [],
    }
    for input_size in config.input_sizes:
        for constellation_size in config.constellation_sizes:
            for split, loader in (
                ("validation", validation_loader),
                ("parameter_ood", ood_loader),
            ):
                curve = _evaluate_curve(
                    refiner,
                    decoder,
                    loader,
                    device,
                    config,
                    input_size=input_size,
                    constellation_size=constellation_size,
                )
                evaluation[split].append(
                    {
                        "input_size": input_size,
                        "constellation_size": constellation_size,
                        "coordinate_payload_bits": 3 * constellation_size * config.bits,
                        "curve": curve,
                    }
                )

    result = {
        "config": asdict(config),
        "device": str(device),
        "decoder_source": decoder_source,
        "decoder_hash_before_refiner": decoder_hash_before,
        "decoder_hash_after_refiner": decoder_hash_after,
        "decoder_unchanged": decoder_unchanged,
        "refiner_hash_before_training": refiner_hash_before,
        "refiner_training_order_hash": refiner_training_order_hash,
        "decoder_parameter_count": sum(
            parameter.numel() for parameter in decoder.parameters()
        ),
        "refiner_parameter_count": sum(
            parameter.numel() for parameter in refiner.parameters()
        ),
        "decoder_history": decoder_history,
        "refiner_history": refiner_history,
        "evaluation": evaluation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(
        {"model": refiner.state_dict(), "config": asdict(config)},
        output_dir / "refiner.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_005_refiner_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--decoder-checkpoint")
    args = parser.parse_args()

    config = RefinerExperimentConfig.from_json(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.decoder_checkpoint is not None:
        config = replace(config, decoder_checkpoint=args.decoder_checkpoint)
    result = run_refiner_experiment(config, device_name=args.device)
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
