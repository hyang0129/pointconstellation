"""Experiment 009: progressive coordinate-only compression homotopy."""

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
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.homotopy import (
    CompressionHomotopyEncoder,
    farthest_point_constellation,
)
from pointconstellation.quantization import quantize_coordinates, quantize_ste
from pointconstellation.train import select_device, set_seed


@dataclass(frozen=True)
class HomotopyExperimentConfig:
    num_points: int = 32
    input_size: int = 32
    stage_sizes: tuple[int, ...] = (16, 8, 4)
    bits: int = 10
    train_samples: int = 14
    validation_samples: int = 7
    parameter_ood_samples: int = 7
    batch_size: int = 7
    decoder_epochs: int = 1
    encoder_epochs_per_stage: int = 1
    decoder_learning_rate: float = 1e-3
    encoder_learning_rate: float = 1e-3
    coverage_weight: float = 0.05
    feature_width: int = 24
    num_heads: int = 4
    num_layers: int = 1
    merge_temperature: float = 0.2
    maximum_update: float = 0.08
    decoder_checkpoint: str | None = None
    seed: int = 7
    output_dir: str = "artifacts/local/experiment_009_homotopy_smoke"

    def __post_init__(self) -> None:
        if self.num_points < 8:
            raise ValueError("num_points must be at least 8")
        if self.input_size < 8 or self.input_size > self.num_points:
            raise ValueError("input_size must be between 8 and num_points")
        if len(self.stage_sizes) < 2:
            raise ValueError("stage_sizes must contain at least two sizes")
        if any(size < 2 for size in self.stage_sizes):
            raise ValueError("every stage size must be at least 2")
        if any(
            first <= second
            for first, second in zip(
                self.stage_sizes, self.stage_sizes[1:], strict=False
            )
        ):
            raise ValueError("stage_sizes must be strictly decreasing")
        if self.stage_sizes[0] > self.input_size:
            raise ValueError("the dense stage cannot exceed input_size")
        if not 2 <= self.bits <= 24:
            raise ValueError("bits must be between 2 and 24")
        if (
            min(
                self.train_samples,
                self.validation_samples,
                self.parameter_ood_samples,
                self.batch_size,
                self.decoder_epochs,
                self.encoder_epochs_per_stage,
            )
            < 1
        ):
            raise ValueError("sample counts, batch size, and epochs must be positive")
        if min(self.decoder_learning_rate, self.encoder_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if self.coverage_weight < 0:
            raise ValueError("coverage_weight cannot be negative")
        if self.feature_width < 4 or self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.merge_temperature <= 0 or self.maximum_update <= 0:
            raise ValueError("temperature and maximum update must be positive")

    @classmethod
    def from_json(cls, path: Path) -> HomotopyExperimentConfig:
        values = json.loads(path.read_text())
        if "stage_sizes" in values:
            values["stage_sizes"] = tuple(values["stage_sizes"])
        return cls(**values)


def _make_loaders(
    config: HomotopyExperimentConfig,
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


def _coverage_loss(constellation: Tensor, target: Tensor) -> Tensor:
    return pairwise_squared(constellation, target).amin(dim=1).mean()


def _train_decoder(
    decoder: VariableConstellationDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: HomotopyExperimentConfig,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.decoder_learning_rate)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.decoder_epochs + 1):
        decoder.train()
        total = 0.0
        count = 0
        for batch in loader:
            target = batch["points"].to(device)
            source = target[:, : config.input_size]
            # Every epoch exposes the shared decoder to every homotopy stage.
            # Cycling one K per batch can silently omit the target K when a
            # smoke dataset contains fewer batches than stages.
            for size in config.stage_sizes:
                constellation = farthest_point_constellation(
                    source, size, config.bits, training=False
                )
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
        record: dict[str, float | int] = {
            "epoch": epoch,
            "training_chamfer_rmse": math.sqrt(total / count),
        }
        history.append(record)
        print(json.dumps({"stage": "decoder", **record}))
    return history


def _encoder_objective(
    encoder: CompressionHomotopyEncoder,
    decoder: VariableConstellationDecoder,
    source: Tensor,
    target: Tensor,
    stage_sizes: tuple[int, ...],
    config: HomotopyExperimentConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    constellation = encoder(source, stage_sizes=stage_sizes)
    assert isinstance(constellation, Tensor)
    reconstruction = decoder(constellation, num_output_points=config.num_points)
    reconstruction_loss = chamfer_squared(reconstruction, target)
    coverage_loss = _coverage_loss(constellation, target)
    loss = reconstruction_loss + config.coverage_weight * coverage_loss
    return loss, reconstruction_loss, coverage_loss


def _train_matched_encoders(
    homotopy: CompressionHomotopyEncoder,
    direct: CompressionHomotopyEncoder,
    decoder: VariableConstellationDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: HomotopyExperimentConfig,
) -> tuple[list[dict[str, Any]], int, int]:
    """Train both encoders on identical batches and equal optimizer-step budgets."""

    decoder.eval().requires_grad_(False)
    homotopy_optimizer = torch.optim.Adam(
        homotopy.parameters(), lr=config.encoder_learning_rate
    )
    direct_optimizer = torch.optim.Adam(
        direct.parameters(), lr=config.encoder_learning_rate
    )
    transitions = len(config.stage_sizes) - 1
    total_epochs = config.encoder_epochs_per_stage * transitions
    target_path = (config.stage_sizes[0], config.stage_sizes[-1])
    history: list[dict[str, Any]] = []
    homotopy_updates = 0
    direct_updates = 0

    for epoch_index in range(total_epochs):
        phase = min(
            epoch_index // config.encoder_epochs_per_stage + 1,
            transitions,
        )
        curriculum_path = config.stage_sizes[: phase + 1]
        homotopy.train()
        direct.train()
        totals = {
            "homotopy": {"loss": 0.0, "reconstruction": 0.0, "coverage": 0.0},
            "direct": {"loss": 0.0, "reconstruction": 0.0, "coverage": 0.0},
        }
        count = 0
        for batch in loader:
            target = batch["points"].to(device)
            source = target[:, : config.input_size]

            homotopy_optimizer.zero_grad(set_to_none=True)
            h_loss, h_reconstruction, h_coverage = _encoder_objective(
                homotopy,
                decoder,
                source,
                target,
                curriculum_path,
                config,
            )
            h_loss.backward()
            homotopy_optimizer.step()
            homotopy_updates += 1

            direct_optimizer.zero_grad(set_to_none=True)
            d_loss, d_reconstruction, d_coverage = _encoder_objective(
                direct,
                decoder,
                source,
                target,
                target_path,
                config,
            )
            d_loss.backward()
            direct_optimizer.step()
            direct_updates += 1

            batch_size = len(target)
            for name, value in (
                ("loss", h_loss),
                ("reconstruction", h_reconstruction),
                ("coverage", h_coverage),
            ):
                totals["homotopy"][name] += float(value.item()) * batch_size
            for name, value in (
                ("loss", d_loss),
                ("reconstruction", d_reconstruction),
                ("coverage", d_coverage),
            ):
                totals["direct"][name] += float(value.item()) * batch_size
            count += batch_size

        record: dict[str, Any] = {
            "epoch": epoch_index + 1,
            "curriculum_path": list(curriculum_path),
        }
        for method in ("homotopy", "direct"):
            means = {name: value / count for name, value in totals[method].items()}
            record[method] = {
                **means,
                "reconstruction_rmse": math.sqrt(means["reconstruction"]),
                "coverage_rmse": math.sqrt(means["coverage"]),
            }
        history.append(record)
        print(json.dumps({"stage": "encoders", **record}))
    return history, homotopy_updates, direct_updates


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
    surface_proxy = anchor_distances.amin(dim=2).mean(dim=1)
    coverage = anchor_distances.amin(dim=1).mean(dim=1)
    return chamfer, surface_proxy, coverage


def _effective_cardinality(constellation: Tensor) -> list[int]:
    return [int(torch.unique(sample, dim=0).shape[0]) for sample in constellation]


def _empty_totals(config: HomotopyExperimentConfig) -> dict[str, dict[int, Any]]:
    return {
        method: {
            size: {
                "chamfer": 0.0,
                "surface_proxy": 0.0,
                "coverage": 0.0,
                "effective_sum": 0,
                "effective_min": size,
            }
            for size in config.stage_sizes
        }
        for method in ("homotopy", "direct", "fps")
    }


def _evaluate(
    homotopy: CompressionHomotopyEncoder,
    direct: CompressionHomotopyEncoder,
    decoder: VariableConstellationDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    config: HomotopyExperimentConfig,
) -> dict[str, Any]:
    homotopy.eval()
    direct.eval()
    decoder.eval()
    totals = _empty_totals(config)
    count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["points"].to(device)
            source = target[:, : config.input_size]
            homotopy_output = homotopy(source, return_history=True)
            assert isinstance(homotopy_output, tuple)
            _, homotopy_history, _ = homotopy_output
            homotopy_states = dict(
                zip(config.stage_sizes, homotopy_history, strict=True)
            )

            direct_states: dict[int, Tensor] = {}
            for size in config.stage_sizes[1:]:
                direct_output = direct(
                    source,
                    stage_sizes=(config.stage_sizes[0], size),
                    return_history=True,
                )
                assert isinstance(direct_output, tuple)
                _, direct_history, _ = direct_output
                direct_states.setdefault(config.stage_sizes[0], direct_history[0])
                direct_states[size] = direct_history[-1]

            fps_states = {
                size: farthest_point_constellation(
                    source, size, config.bits, training=False
                )
                for size in config.stage_sizes
            }

            for method, states in (
                ("homotopy", homotopy_states),
                ("direct", direct_states),
                ("fps", fps_states),
            ):
                for size, raw_constellation in states.items():
                    constellation = quantize_coordinates(raw_constellation, config.bits)
                    reconstruction = decoder(
                        constellation, num_output_points=config.num_points
                    )
                    chamfer, surface_proxy, coverage = _sample_metrics(
                        reconstruction, target, constellation
                    )
                    stage_totals = totals[method][size]
                    stage_totals["chamfer"] += float(chamfer.sum().item())
                    stage_totals["surface_proxy"] += float(surface_proxy.sum().item())
                    stage_totals["coverage"] += float(coverage.sum().item())
                    effective = _effective_cardinality(constellation)
                    stage_totals["effective_sum"] += sum(effective)
                    stage_totals["effective_min"] = min(
                        stage_totals["effective_min"], min(effective)
                    )
            count += len(target)

    results: dict[str, list[dict[str, Any]]] = {}
    for method in ("homotopy", "direct", "fps"):
        stages = []
        for size in config.stage_sizes:
            values = totals[method][size]
            chamfer = values["chamfer"] / count
            surface_proxy = values["surface_proxy"] / count
            coverage = values["coverage"] / count
            stages.append(
                {
                    "requested_k": size,
                    "coordinate_payload_bits": 3 * size * config.bits,
                    "exact_quantized": True,
                    "chamfer": chamfer,
                    "chamfer_rmse": math.sqrt(chamfer),
                    "surface_proxy": surface_proxy,
                    "surface_proxy_rmse": math.sqrt(surface_proxy),
                    "coverage": coverage,
                    "coverage_rmse": math.sqrt(coverage),
                    "effective_k_mean": values["effective_sum"] / count,
                    "effective_k_min": values["effective_min"],
                    "trained_for_requested_k": (
                        method != "direct"
                        or size in (config.stage_sizes[0], config.stage_sizes[-1])
                    ),
                }
            )
        results[method] = stages

    homotopy_target = results["homotopy"][-1]["chamfer_rmse"]
    direct_target = results["direct"][-1]["chamfer_rmse"]
    fps_target = results["fps"][-1]["chamfer_rmse"]
    results["target_k_gap"] = {
        "requested_k": config.stage_sizes[-1],
        "homotopy_chamfer_rmse": homotopy_target,
        "direct_chamfer_rmse": direct_target,
        "homotopy_minus_direct_chamfer_rmse": homotopy_target - direct_target,
        "homotopy_improvement_percent": (
            100.0 * (direct_target - homotopy_target) / max(direct_target, 1e-12)
        ),
        "fps_chamfer_rmse": fps_target,
        "homotopy_gap_vs_fps_percent": (
            100.0 * (homotopy_target - fps_target) / max(fps_target, 1e-12)
        ),
        "direct_gap_vs_fps_percent": (
            100.0 * (direct_target - fps_target) / max(fps_target, 1e-12)
        ),
    }
    return results


def run_homotopy_experiment(
    config: HomotopyExperimentConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train a matched direct/homotopy comparison and write reproducible artifacts."""

    set_seed(config.seed)
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, ood_loader = _make_loaders(config)
    decoder = VariableConstellationDecoder(
        config.num_points,
        config.stage_sizes[0],
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

    # Construct once, then copy: this is a bitwise matched initialization. Both
    # optimizers also receive the same number of updates over identical batches.
    torch.manual_seed(config.seed + 9009)
    base_encoder = CompressionHomotopyEncoder(
        config.stage_sizes,
        bits=config.bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        merge_temperature=config.merge_temperature,
        maximum_update=config.maximum_update,
    )
    homotopy = copy.deepcopy(base_encoder).to(device)
    direct = copy.deepcopy(base_encoder).to(device)
    homotopy_initial_hash = _state_hash(homotopy)
    direct_initial_hash = _state_hash(direct)
    initial_parameters_matched = homotopy_initial_hash == direct_initial_hash
    if not initial_parameters_matched:
        raise RuntimeError("direct and homotopy initializations do not match")

    decoder.eval().requires_grad_(False)
    decoder_hash_before = _state_hash(decoder)
    encoder_history, homotopy_updates, direct_updates = _train_matched_encoders(
        homotopy,
        direct,
        decoder,
        train_loader,
        device,
        config,
    )
    decoder_hash_after_training = _state_hash(decoder)
    if homotopy_updates != direct_updates:
        raise RuntimeError("matched encoders received different update counts")

    evaluation = {
        "validation": _evaluate(
            homotopy, direct, decoder, validation_loader, device, config
        ),
        "parameter_ood": _evaluate(
            homotopy, direct, decoder, ood_loader, device, config
        ),
    }
    decoder_hash_final = _state_hash(decoder)
    decoder_unchanged = (
        decoder_hash_before == decoder_hash_after_training == decoder_hash_final
    )
    if not decoder_unchanged:
        raise RuntimeError("frozen decoder changed during encoder training/evaluation")

    result = {
        "experiment": "009_compression_homotopy",
        "config": asdict(config),
        "device": str(device),
        "decoder_source": decoder_source,
        "decoder_history": decoder_history,
        "decoder_hash_before_encoders": decoder_hash_before,
        "decoder_hash_after_training": decoder_hash_after_training,
        "decoder_hash_final": decoder_hash_final,
        "decoder_unchanged": decoder_unchanged,
        "homotopy_initial_hash": homotopy_initial_hash,
        "direct_initial_hash": direct_initial_hash,
        "initial_parameters_matched": initial_parameters_matched,
        "homotopy_updates": homotopy_updates,
        "direct_updates": direct_updates,
        "update_budget_matched": homotopy_updates == direct_updates,
        "decoder_parameter_count": sum(
            parameter.numel() for parameter in decoder.parameters()
        ),
        "encoder_parameter_count": sum(
            parameter.numel() for parameter in homotopy.parameters()
        ),
        "encoder_history": encoder_history,
        "evaluation": evaluation,
        "elapsed_seconds": time.perf_counter() - started,
        "caveats": [
            "Surface proxy is distance to finite observed samples, not an "
            "analytic surface.",
            "Equal optimizer-step budgets do not imply equal compute: homotopy "
            "uses more transitions.",
            "FPS and hard assignments are deterministic except at exact "
            "geometric ties.",
        ],
    }
    torch.save(
        {"model": homotopy.state_dict(), "config": asdict(config)},
        output_dir / "homotopy_encoder.pt",
    )
    torch.save(
        {"model": direct.state_dict(), "config": asdict(config)},
        output_dir / "direct_encoder.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_009_homotopy_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--decoder-checkpoint")
    args = parser.parse_args()

    config = HomotopyExperimentConfig.from_json(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.decoder_checkpoint is not None:
        config = replace(config, decoder_checkpoint=args.decoder_checkpoint)
    result = run_homotopy_experiment(config, device_name=args.device)
    print(
        json.dumps(
            {
                "device": result["device"],
                "decoder_unchanged": result["decoder_unchanged"],
                "update_budget_matched": result["update_budget_matched"],
                "validation_target_k_gap": result["evaluation"]["validation"][
                    "target_k_gap"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
