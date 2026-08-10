"""Train the first coordinate-only constellation autoencoder."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pointconstellation.data import ProceduralPointCloudDataset
from pointconstellation.losses import constellation_loss
from pointconstellation.models import ConstellationAutoencoder


@dataclass(frozen=True)
class TrainingConfig:
    num_points: int = 256
    constellation_size: int = 16
    bits: int = 12
    train_samples: int = 224
    validation_samples: int = 70
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 1e-3
    surface_weight: float = 0.1
    repulsion_weight: float = 0.01
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_001_smoke"

    @classmethod
    def from_json(cls, path: Path) -> TrainingConfig:
        return cls(**json.loads(path.read_text()))


def select_device(requested: str = "auto") -> torch.device:
    """Resolve CUDA, Apple MPS, or CPU in that priority order."""

    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _accumulate(
    totals: dict[str, float], components: dict[str, Tensor], batch_size: int
) -> None:
    for name, value in components.items():
        totals[name] = totals.get(name, 0.0) + float(value.item()) * batch_size


def run_epoch(
    model: ConstellationAutoencoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: TrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        points = batch["points"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction, constellation = model(points)
            loss, components = constellation_loss(
                reconstruction,
                points,
                constellation,
                surface_weight=config.surface_weight,
                repulsion_weight=config.repulsion_weight,
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_size = len(points)
        _accumulate(totals, components, batch_size)
        count += batch_size
    metrics = {name: value / count for name, value in totals.items()}
    metrics["chamfer_rmse"] = metrics["chamfer"] ** 0.5
    return metrics


def train(
    config: TrainingConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ProceduralPointCloudDataset(
        config.train_samples,
        num_points=config.num_points,
        seed=config.seed,
        split="train",
    )
    validation_dataset = ProceduralPointCloudDataset(
        config.validation_samples,
        num_points=config.num_points,
        seed=config.seed,
        split="validation",
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = ConstellationAutoencoder(
        num_input_points=config.num_points,
        constellation_size=config.constellation_size,
        bits=config.bits,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    started = time.perf_counter()
    history = []
    initial = run_epoch(model, validation_loader, device, config, optimizer=None)
    print(json.dumps({"epoch": 0, "validation": initial, "device": str(device)}))
    for epoch in range(1, config.epochs + 1):
        training = run_epoch(model, train_loader, device, config, optimizer=optimizer)
        validation = run_epoch(model, validation_loader, device, config, optimizer=None)
        record = {"epoch": epoch, "train": training, "validation": validation}
        history.append(record)
        print(json.dumps(record))

    result = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "coordinate_payload_bits": 3 * config.constellation_size * config.bits,
        "initial_validation": initial,
        "final_validation": history[-1]["validation"],
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    torch.save(
        {"model": model.state_dict(), "config": asdict(config)},
        output_dir / "model.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_001_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    values = asdict(TrainingConfig.from_json(args.config))
    if args.epochs is not None:
        values["epochs"] = args.epochs
    if args.output_dir is not None:
        values["output_dir"] = args.output_dir
    result = train(TrainingConfig(**values), device_name=args.device)
    print(
        json.dumps({key: result[key] for key in result if key != "history"}, indent=2)
    )


if __name__ == "__main__":
    main()
