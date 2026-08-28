"""Experiment 019: decompose decoder/refiner seed instability.

This runner deliberately studies one fixed coordinate rate.  Every per-cloud
message is serialized as an unordered, quantized ``K x 3`` coordinate set.  A
decoder is trained once for each decoder seed, then held fixed while independent
refiners are crossed with it.  The stabilized decoder arm is selected using
only a calibration holdout from the training split (or an explicit manifest
``calibration`` split); validation and OOD clouds are never consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from pointconstellation.bitstream import (
    HEADER,
    MODE_ENTROPY,
    MODE_FIXED,
    ConstellationPacket,
    decode_constellation,
    encode_constellation,
    entropy_bound_bytes,
    expected_stream_bytes,
)
from pointconstellation.data import (
    MeshSurfaceDataset,
    ProceduralPointCloudDataset,
    RawPointCloudDataset,
    file_sha256,
    load_mesh_manifest,
    load_pointcloud_manifest,
)
from pointconstellation.losses import pairwise_squared
from pointconstellation.metrics import point_set_error_metrics
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.gradient_free import adam_ste_search
from pointconstellation.models.refiner import CompetitiveConstellationRefiner
from pointconstellation.quantization import quantize_ste
from pointconstellation.rate_accounting import model_amortization
from pointconstellation.refiner_experiment import _fps, _state_hash
from pointconstellation.surface_metrics import mesh_surface_metrics
from pointconstellation.train import select_device, set_seed

ARMS = ("baseline", "stabilized")
EVALUATION_SPLITS = ("validation", "ood")


@dataclass(frozen=True)
class StabilityExperimentConfig:
    """Configuration for the crossed decoder/refiner stability experiment."""

    dataset_kind: str = "procedural"
    dataset_root: str | None = None
    dataset_manifest: str | None = None
    calibration_split: str = "train"
    mesh_ood_split: str = "category_ood"
    verify_mesh_hashes: bool = True
    verify_pointcloud_hashes: bool = True
    pointcloud_normal_neighbors: int = 16
    num_points: int = 2048
    constellation_size: int = 8
    training_constellation_sizes: tuple[int, ...] = (4, 8, 16, 32)
    coordinate_bits: int = 12
    train_samples: int = 480
    calibration_samples: int = 32
    validation_samples: int = 128
    ood_samples: int = 32
    batch_size: int = 4
    decoder_seeds: tuple[int, ...] = (7, 17, 29, 41, 53, 67)
    refiner_seeds: tuple[int, ...] = (101, 211, 307)
    data_seed: int = 1517
    baseline_decoder_epochs: int = 1
    stabilized_decoder_epochs: int = 4
    decoder_learning_rate: float = 1e-3
    ema_decay: float = 0.99
    refiner_epochs: int = 2
    refiner_learning_rate: float = 1e-3
    feature_width: int = 32
    num_heads: int = 4
    num_layers: int = 1
    recurrent_steps: int = 2
    responsibility_temperature: float = 0.2
    maximum_update: float = 0.1
    use_decoder_gradient: bool = True
    distance_chunk_size: int = 256
    compute_mesh_metrics: bool = False
    mesh_metric_point_chunk_size: int = 256
    mesh_metric_triangle_chunk_size: int = 256
    mesh_normal_neighbors: int = 12
    adam_probe_evaluations: int = 16
    adam_probe_learning_rate: float = 0.03
    adam_probe_clouds_per_split: int = 16
    reference_feature_dir: str | None = None
    expected_manifest_sha256: str | None = None
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 20_260_816
    confidence_level: float = 0.95
    output_dir: str = "artifacts/local/experiment_019_stability"

    def __post_init__(self) -> None:
        if self.dataset_kind not in {
            "procedural",
            "mesh_manifest",
            "pointcloud_manifest",
        }:
            raise ValueError(
                "dataset_kind must be procedural, mesh_manifest, or pointcloud_manifest"
            )
        if self.dataset_kind in {"mesh_manifest", "pointcloud_manifest"} and (
            self.dataset_root is None or self.dataset_manifest is None
        ):
            raise ValueError("manifest datasets require root and manifest")
        # Keep checkpoint selection structurally unable to read a test split.
        if self.calibration_split not in {"train", "calibration"}:
            raise ValueError("calibration_split must be train or calibration")
        if self.dataset_kind == "procedural" and self.calibration_split != "train":
            raise ValueError("procedural calibration must be held out from train")
        if self.num_points < 8:
            raise ValueError("num_points must be at least 8")
        if self.pointcloud_normal_neighbors < 3:
            raise ValueError("pointcloud_normal_neighbors must be at least 3")
        if not 2 <= self.constellation_size <= self.num_points:
            raise ValueError("constellation_size must be between 2 and num_points")
        if (
            not self.training_constellation_sizes
            or len(set(self.training_constellation_sizes))
            != len(self.training_constellation_sizes)
            or self.constellation_size not in self.training_constellation_sizes
            or min(self.training_constellation_sizes) < 2
            or max(self.training_constellation_sizes) > self.num_points
        ):
            raise ValueError(
                "training_constellation_sizes must be unique, valid, and include K"
            )
        if not 2 <= self.coordinate_bits <= 24:
            raise ValueError("coordinate_bits must be between 2 and 24")
        counts = (
            self.train_samples,
            self.calibration_samples,
            self.validation_samples,
            self.ood_samples,
            self.batch_size,
            self.baseline_decoder_epochs,
            self.stabilized_decoder_epochs,
            self.refiner_epochs,
            self.distance_chunk_size,
            self.mesh_metric_point_chunk_size,
            self.mesh_metric_triangle_chunk_size,
            self.mesh_normal_neighbors,
        )
        if min(counts) < 1:
            raise ValueError("sample, batch, epoch, and chunk counts must be positive")
        if self.mesh_normal_neighbors < 3:
            raise ValueError("mesh_normal_neighbors must be at least three")
        if self.stabilized_decoder_epochs <= self.baseline_decoder_epochs:
            raise ValueError("stabilized decoder training must be longer than baseline")
        if len(self.decoder_seeds) < 2 or len(set(self.decoder_seeds)) != len(
            self.decoder_seeds
        ):
            raise ValueError("decoder_seeds must contain at least two unique seeds")
        if len(self.refiner_seeds) < 2 or len(set(self.refiner_seeds)) != len(
            self.refiner_seeds
        ):
            raise ValueError("refiner_seeds must contain at least two unique seeds")
        if self.feature_width < 4 or self.feature_width % self.num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if self.num_layers < 1 or self.recurrent_steps < 1:
            raise ValueError("model layer and recurrent step counts must be positive")
        if min(self.decoder_learning_rate, self.refiner_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if self.responsibility_temperature <= 0 or self.maximum_update <= 0:
            raise ValueError("refiner temperature and update must be positive")
        if self.adam_probe_evaluations < 0 or self.adam_probe_clouds_per_split < 0:
            raise ValueError("Adam probe counts cannot be negative")
        if self.adam_probe_evaluations and self.adam_probe_learning_rate <= 0:
            raise ValueError("Adam probe learning rate must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

    @classmethod
    def from_json(cls, path: Path) -> StabilityExperimentConfig:
        values = json.loads(path.read_text())
        for key in (
            "decoder_seeds",
            "refiner_seeds",
            "training_constellation_sizes",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


DatasetMap = dict[str, Dataset[dict[str, Any]]]


def stability_config_matches_artifact(
    artifact_config: Mapping[str, Any], config: StabilityExperimentConfig
) -> bool:
    """Compare configs while accepting absent default Experiment 031 fields."""

    actual = dict(artifact_config)
    expected = json.loads(json.dumps(asdict(config)))
    for field in (
        "compute_mesh_metrics",
        "mesh_metric_point_chunk_size",
        "mesh_metric_triangle_chunk_size",
        "mesh_normal_neighbors",
    ):
        actual.setdefault(field, expected[field])
    return actual == expected


def _mesh_dataset(config: StabilityExperimentConfig, split: str) -> MeshSurfaceDataset:
    assert config.dataset_root is not None and config.dataset_manifest is not None
    return MeshSurfaceDataset(
        Path(config.dataset_root),
        Path(config.dataset_manifest),
        split=split,
        num_points=config.num_points,
        seed=config.data_seed,
        verify_hashes=config.verify_mesh_hashes,
        training_target="source",
    )


def _pointcloud_dataset(
    config: StabilityExperimentConfig, split: str
) -> RawPointCloudDataset:
    assert config.dataset_root is not None and config.dataset_manifest is not None
    return RawPointCloudDataset(
        Path(config.dataset_root),
        Path(config.dataset_manifest),
        split=split,
        num_points=config.num_points,
        seed=config.data_seed,
        normal_neighbors=config.pointcloud_normal_neighbors,
        verify_hashes=config.verify_pointcloud_hashes,
    )


def _datasets(config: StabilityExperimentConfig) -> DatasetMap:
    """Build disjoint train/calibration data and untouched evaluation splits."""

    if config.dataset_kind == "procedural":
        combined = ProceduralPointCloudDataset(
            config.train_samples + config.calibration_samples,
            num_points=config.num_points,
            seed=config.data_seed,
            split="train",
        )
        return {
            "train": Subset(combined, range(config.train_samples)),
            "calibration": Subset(
                combined,
                range(
                    config.train_samples,
                    config.train_samples + config.calibration_samples,
                ),
            ),
            "validation": ProceduralPointCloudDataset(
                config.validation_samples,
                num_points=config.num_points,
                seed=config.data_seed,
                split="validation",
            ),
            "ood": ProceduralPointCloudDataset(
                config.ood_samples,
                num_points=config.num_points,
                seed=config.data_seed,
                split="parameter_ood",
            ),
        }

    dataset_factory = (
        _mesh_dataset if config.dataset_kind == "mesh_manifest" else _pointcloud_dataset
    )
    train = dataset_factory(config, "train")
    if config.calibration_split == "train":
        expected = config.train_samples + config.calibration_samples
        if len(train) != expected:
            raise ValueError(
                f"manifest train has {len(train)} records; expected {expected} "
                "for the declared train/calibration holdout"
            )
        training: Dataset[dict[str, Any]] = Subset(train, range(config.train_samples))
        calibration: Dataset[dict[str, Any]] = Subset(
            train,
            range(config.train_samples, expected),
        )
    else:
        if len(train) != config.train_samples:
            raise ValueError(
                f"manifest train has {len(train)} records; expected "
                f"{config.train_samples}"
            )
        calibration = dataset_factory(config, "calibration")
        training = train
        if len(calibration) != config.calibration_samples:
            raise ValueError(
                f"manifest calibration has {len(calibration)} records; expected "
                f"{config.calibration_samples}"
            )
    validation = dataset_factory(config, "validation")
    ood = dataset_factory(config, config.mesh_ood_split)
    expected_evaluation = {
        "validation": (validation, config.validation_samples),
        "ood": (ood, config.ood_samples),
    }
    for name, (dataset, expected) in expected_evaluation.items():
        if len(dataset) != expected:
            raise ValueError(
                f"manifest {name} has {len(dataset)} records; expected {expected}"
            )
    return {
        "train": training,
        "calibration": calibration,
        "validation": validation,
        "ood": ood,
    }


def _membership_records(dataset: Dataset[dict[str, Any]]) -> list[str]:
    if isinstance(dataset, Subset):
        parent = dataset.dataset
        if isinstance(parent, (MeshSurfaceDataset, RawPointCloudDataset)):
            return [
                f"{parent.records[index]['category']}:"
                f"{parent.records[index]['model_id']}"
                for index in dataset.indices
            ]
        if isinstance(parent, ProceduralPointCloudDataset):
            return [f"{parent.split}:{index}" for index in dataset.indices]
    if isinstance(dataset, (MeshSurfaceDataset, RawPointCloudDataset)):
        return [
            f"{record['category']}:{record['model_id']}" for record in dataset.records
        ]
    if isinstance(dataset, ProceduralPointCloudDataset):
        return [f"{dataset.split}:{index}" for index in range(len(dataset))]
    raise TypeError(f"unsupported dataset type for membership: {type(dataset)}")


def _membership(dataset: Dataset[dict[str, Any]]) -> dict[str, Any]:
    records = _membership_records(dataset)
    digest = hashlib.sha256("\n".join(records).encode()).hexdigest()
    return {"count": len(records), "sha256": digest, "records": records}


def _data_protocol(
    config: StabilityExperimentConfig, datasets: DatasetMap
) -> dict[str, Any]:
    partitions = {name: _membership(dataset) for name, dataset in datasets.items()}
    record_sets = {
        name: set(partition["records"]) for name, partition in partitions.items()
    }
    disjoint = {
        f"{left}_vs_{right}": not bool(record_sets[left] & record_sets[right])
        for index, left in enumerate(partitions)
        for right in tuple(partitions)[index + 1 :]
    }
    identity: dict[str, Any] = {
        "dataset_kind": config.dataset_kind,
        "data_seed": config.data_seed,
        "calibration_source": (
            "held_out_train_records"
            if config.calibration_split == "train"
            else "explicit_calibration_manifest_split"
        ),
        "calibration_forbidden_splits": ["validation", "ood"],
        "partitions": partitions,
        "all_partitions_pairwise_disjoint": all(disjoint.values()),
        "pairwise_disjoint": disjoint,
    }
    if config.dataset_kind in {"mesh_manifest", "pointcloud_manifest"}:
        assert config.dataset_manifest is not None
        manifest_path = Path(config.dataset_manifest)
        manifest = (
            load_mesh_manifest(manifest_path)
            if config.dataset_kind == "mesh_manifest"
            else load_pointcloud_manifest(manifest_path)
        )
        manifest_sha256 = file_sha256(manifest_path)
        if (
            config.expected_manifest_sha256 is not None
            and manifest_sha256 != config.expected_manifest_sha256
        ):
            raise ValueError(
                "dataset manifest hash differs from expected_manifest_sha256"
            )
        train_records = manifest["splits"]["train"]
        calibration_records = manifest["splits"][config.calibration_split]
        validation_records = manifest["splits"]["validation"]
        ood_records = manifest["splits"][config.mesh_ood_split]
        all_records = (
            *train_records,
            *calibration_records,
            *validation_records,
            *ood_records,
        )
        has_official_splits = any("official_split" in record for record in all_records)
        if has_official_splits:
            if any(
                record.get("official_split") != "train"
                for record in (*train_records, *calibration_records)
            ):
                raise ValueError(
                    "train/calibration records must be official training data"
                )
            if any(
                record.get("official_split") != "test"
                for record in (*validation_records, *ood_records)
            ):
                raise ValueError("validation/OOD records must be official test data")
        elif any(
            record.get("manifest_role") != split
            for split, records in (
                ("train", train_records),
                (config.calibration_split, calibration_records),
                ("validation", validation_records),
                (config.mesh_ood_split, ood_records),
            )
            for record in records
        ):
            raise ValueError("manifest record role differs from its declared split")
        trained_categories = {record["category"] for record in train_records}
        calibration_categories = {record["category"] for record in calibration_records}
        validation_categories = {record["category"] for record in validation_records}
        heldout_categories = {record["category"] for record in ood_records}
        if not calibration_categories <= trained_categories:
            raise ValueError("calibration includes a category absent from training")
        if not validation_categories <= trained_categories:
            raise ValueError("validation includes a category absent from training")
        if (calibration_categories | validation_categories) & heldout_categories:
            raise ValueError("an ID evaluation role includes a held-out OOD category")
        identity.update(
            {
                "dataset": manifest["dataset"],
                "manifest_sha256": manifest_sha256,
                "mesh_ood_split": config.mesh_ood_split,
                "verify_hashes": (
                    config.verify_mesh_hashes
                    if config.dataset_kind == "mesh_manifest"
                    else config.verify_pointcloud_hashes
                ),
                "verify_mesh_hashes": (
                    config.verify_mesh_hashes
                    if config.dataset_kind == "mesh_manifest"
                    else None
                ),
                "verify_pointcloud_hashes": (
                    config.verify_pointcloud_hashes
                    if config.dataset_kind == "pointcloud_manifest"
                    else None
                ),
                "official_split_checks": {
                    "applicable": has_official_splits,
                    "train_and_calibration_are_official_train": has_official_splits,
                    "validation_and_ood_are_official_test": has_official_splits,
                    "calibration_categories_exclude_heldout": True,
                    "validation_categories_exclude_heldout": True,
                },
                "normals_estimated": manifest.get("sampling", {}).get(
                    "normals_estimated", False
                ),
            }
        )
    return identity


def _loader(
    dataset: Dataset[dict[str, Any]],
    *,
    config: StabilityExperimentConfig,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _source(batch: Mapping[str, Any], device: torch.device) -> Tensor:
    key = "source_points" if "source_points" in batch else "points"
    return batch[key].to(device)


def _fresh_target(batch: Mapping[str, Any], source: Tensor) -> Tensor:
    if "fresh_points" in batch:
        return batch["fresh_points"].to(source.device)
    return source


def _per_cloud_chamfer(first: Tensor, second: Tensor, *, chunk_size: int) -> Tensor:
    def directed(source: Tensor, target: Tensor) -> Tensor:
        minima = []
        for start in range(0, source.shape[1], chunk_size):
            distances = pairwise_squared(source[:, start : start + chunk_size], target)
            minima.append(distances.amin(dim=2))
        return torch.cat(minima, dim=1).mean(dim=1)

    return 0.5 * (directed(first, second) + directed(second, first))


def _decoder(config: StabilityExperimentConfig, device: torch.device) -> nn.Module:
    return VariableConstellationDecoder(
        config.num_points,
        max(config.training_constellation_sizes),
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    ).to(device)


def _refiner(config: StabilityExperimentConfig, device: torch.device) -> nn.Module:
    return CompetitiveConstellationRefiner(
        max(config.training_constellation_sizes),
        bits=config.coordinate_bits,
        feature_width=config.feature_width,
        num_heads=config.num_heads,
        recurrent_steps=config.recurrent_steps,
        responsibility_temperature=config.responsibility_temperature,
        maximum_update=config.maximum_update,
        use_decoder_gradient=config.use_decoder_gradient,
        decoder_gradient_chunk_size=config.distance_chunk_size,
    ).to(device)


def _clone_state(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _update_ema(ema: dict[str, Tensor], module: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in module.state_dict().items():
            current = value.detach().cpu()
            if current.is_floating_point():
                ema[name].mul_(decay).add_(current, alpha=1.0 - decay)
            else:
                ema[name].copy_(current)


def _ema_calibration_candidate(
    module: nn.Module,
    ema: dict[str, Tensor],
    *,
    epoch: int,
    calibration_score: Callable[[nn.Module], float],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Score an EMA state without changing the continuing raw training state."""

    raw_state = _clone_state(module)
    try:
        module.load_state_dict(ema)
        score = calibration_score(module)
        candidate_state = _clone_state(module)
        candidate_hash = _state_hash(module)
    finally:
        module.load_state_dict(raw_state)
    return candidate_state, {
        "epoch": epoch,
        "kind": "ema",
        "calibration_chamfer_rmse": score,
        "state_hash": candidate_hash,
    }


def _calibration_candidate_is_better(
    candidate: Mapping[str, Any], selected: Mapping[str, Any] | None
) -> bool:
    """Use the predeclared calibration objective with stable earliest-epoch ties."""

    return selected is None or (
        candidate["calibration_chamfer_rmse"] < selected["calibration_chamfer_rmse"]
    )


def _calibration_rmse(
    decoder: nn.Module,
    dataset: Dataset[dict[str, Any]],
    *,
    config: StabilityExperimentConfig,
    device: torch.device,
) -> float:
    decoder.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in _loader(dataset, config=config, shuffle=False):
            source = _source(batch, device)
            constellation = _fps(
                source, config.constellation_size, config.coordinate_bits
            )
            constellation, _, exact, lattice_exact = _serialized_coordinates(
                constellation, config=config
            )
            if not exact or not lattice_exact:
                raise RuntimeError("calibration FPS bitstream round trip failed")
            reconstruction = decoder(constellation, num_output_points=config.num_points)
            losses = _per_cloud_chamfer(
                reconstruction, source, chunk_size=config.distance_chunk_size
            )
            total += float(losses.sum().item())
            count += len(source)
    return math.sqrt(total / count)


def _train_decoders(
    config: StabilityExperimentConfig,
    datasets: DatasetMap,
    *,
    decoder_seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, dict[str, Tensor]], dict[str, Any]]:
    """Train a matched baseline and longer EMA-selected stabilized decoder."""

    set_seed(decoder_seed)
    decoder = _decoder(config, device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.decoder_learning_rate)
    ema = _clone_state(decoder)
    loader = _loader(datasets["train"], config=config, shuffle=True, seed=decoder_seed)
    history: list[dict[str, Any]] = []
    baseline_state: dict[str, Tensor] | None = None
    baseline_calibration_rmse: float | None = None
    stabilized_state: dict[str, Tensor] | None = None
    stabilized_choice: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    started = time.perf_counter()
    global_step = 0

    for epoch in range(1, config.stabilized_decoder_epochs + 1):
        decoder.train()
        total = 0.0
        count = 0
        for batch in loader:
            source = _source(batch, device)
            training_size = config.training_constellation_sizes[
                global_step % len(config.training_constellation_sizes)
            ]
            constellation = _fps(source, training_size, config.coordinate_bits)
            constellation = quantize_ste(
                constellation,
                config.coordinate_bits,
                training=True,
                jitter=True,
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction = decoder(constellation, num_output_points=config.num_points)
            losses = _per_cloud_chamfer(
                reconstruction, source, chunk_size=config.distance_chunk_size
            )
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            _update_ema(ema, decoder, config.ema_decay)
            total += float(losses.detach().sum().item())
            count += len(source)
            global_step += 1

        record: dict[str, Any] = {
            "epoch": epoch,
            "training_chamfer_rmse": math.sqrt(total / count),
        }
        if epoch == config.baseline_decoder_epochs:
            baseline_state = _clone_state(decoder)
            baseline_calibration_rmse = _calibration_rmse(
                decoder,
                datasets["calibration"],
                config=config,
                device=device,
            )
            record["baseline_calibration_chamfer_rmse"] = baseline_calibration_rmse
        if epoch > config.baseline_decoder_epochs:
            candidate_state, candidate = _ema_calibration_candidate(
                decoder,
                ema,
                epoch=epoch,
                calibration_score=lambda candidate_decoder: _calibration_rmse(
                    candidate_decoder,
                    datasets["calibration"],
                    config=config,
                    device=device,
                ),
            )
            candidates.append(candidate)
            record["ema_calibration_chamfer_rmse"] = candidate[
                "calibration_chamfer_rmse"
            ]
            if _calibration_candidate_is_better(candidate, stabilized_choice):
                stabilized_choice = candidate
                stabilized_state = candidate_state
        history.append(record)
        print(
            json.dumps(
                {
                    "stage": "decoder",
                    "decoder_seed": decoder_seed,
                    **record,
                }
            ),
            flush=True,
        )

    if baseline_state is None or baseline_calibration_rmse is None:
        raise RuntimeError("baseline decoder checkpoint was not captured")
    if stabilized_state is None or stabilized_choice is None:
        raise RuntimeError("no stabilized EMA checkpoint was eligible")

    arm_states = {"baseline": baseline_state, "stabilized": stabilized_state}
    decoder_dir = output_dir / "decoders" / f"seed_{decoder_seed}"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    arm_records: dict[str, dict[str, Any]] = {}
    for arm, state in arm_states.items():
        decoder.load_state_dict(state)
        state_hash = _state_hash(decoder)
        selection = (
            {
                "kind": "raw_fixed_epoch",
                "epoch": config.baseline_decoder_epochs,
                "calibration_chamfer_rmse": baseline_calibration_rmse,
            }
            if arm == "baseline"
            else stabilized_choice
        )
        selection = {**selection, "selected_state_hash": state_hash}
        checkpoint = decoder_dir / f"{arm}.pt"
        torch.save(
            {
                "model": state,
                "decoder_seed": decoder_seed,
                "arm": arm,
                "state_hash": state_hash,
                "selection": selection,
            },
            checkpoint,
        )
        deployment_files = {}
        for precision, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            deployment_path = decoder_dir / arm / f"decoder_state_dict_{precision}.pt"
            deployment_path.parent.mkdir(exist_ok=True)
            torch.save(
                {
                    name: (
                        value.detach().cpu().to(dtype)
                        if value.is_floating_point()
                        else value.detach().cpu()
                    )
                    for name, value in state.items()
                },
                deployment_path,
            )
            deployment_files[precision] = {
                "path": str(deployment_path),
                "bytes": deployment_path.stat().st_size,
                "sha256": file_sha256(deployment_path),
            }
        arm_records[arm] = {
            "state_hash": state_hash,
            "checkpoint": str(checkpoint),
            "selection": selection,
            "decoder_state_dicts": deployment_files,
        }
    record = {
        "decoder_seed": decoder_seed,
        "history": history,
        "ema_decay": config.ema_decay,
        "stabilized_candidates": candidates,
        "arms": arm_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (decoder_dir / "training_metrics.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    # This file is written before any validation/OOD loader is constructed.
    # It seals the complete calibration candidate set and the chosen hashes.
    (decoder_dir / "selection.json").write_text(
        json.dumps(
            {
                "decoder_seed": decoder_seed,
                "calibration_partition_sha256": _membership(datasets["calibration"])[
                    "sha256"
                ],
                "selection_metric": (
                    "K8-equivalent exact serialized FPS source-cloud aggregate "
                    "Chamfer RMSE"
                ),
                "primary_constellation_size": config.constellation_size,
                "coordinate_bits": config.coordinate_bits,
                "expected_stream_bytes": expected_stream_bytes(
                    config.constellation_size, config.coordinate_bits
                ),
                "stabilized_candidates": candidates,
                "arms": arm_records,
            },
            indent=2,
        )
        + "\n"
    )
    return arm_states, record


def _train_refiner(
    config: StabilityExperimentConfig,
    datasets: DatasetMap,
    *,
    decoder: nn.Module,
    refiner_seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    decoder.eval().requires_grad_(False)
    decoder_hash_before = _state_hash(decoder)
    # Setting the seed after decoder construction makes the same refiner seed
    # share initialization and batch order across every decoder/arm cell.
    set_seed(refiner_seed)
    refiner = _refiner(config, device)
    initial_hash = _state_hash(refiner)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=config.refiner_learning_rate)
    loader = _loader(datasets["train"], config=config, shuffle=True, seed=refiner_seed)
    history: list[dict[str, Any]] = []
    order_digest = hashlib.sha256()
    started = time.perf_counter()
    global_step = 0
    for epoch in range(1, config.refiner_epochs + 1):
        refiner.train()
        total = 0.0
        count = 0
        for batch in loader:
            sample_ids = batch["sample_id"]
            if isinstance(sample_ids, Tensor):
                order_digest.update(sample_ids.cpu().numpy().tobytes())
            else:
                order_digest.update(repr(tuple(sample_ids)).encode())
            source = _source(batch, device)
            training_size = config.training_constellation_sizes[
                global_step % len(config.training_constellation_sizes)
            ]
            optimizer.zero_grad(set_to_none=True)
            _, states = refiner(
                source,
                training_size,
                decoder=decoder,
                target=source,
                num_output_points=config.num_points,
                return_history=True,
            )
            state_losses = [
                _per_cloud_chamfer(
                    decoder(state, num_output_points=config.num_points),
                    source,
                    chunk_size=config.distance_chunk_size,
                ).mean()
                for state in states[1:]
            ]
            loss = torch.stack(state_losses).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item()) * len(source)
            count += len(source)
            global_step += 1
        record = {
            "epoch": epoch,
            "training_refined_chamfer_rmse": math.sqrt(total / count),
        }
        history.append(record)
    decoder_hash_after = _state_hash(decoder)
    if decoder_hash_before != decoder_hash_after:
        raise RuntimeError("frozen decoder changed during refiner training")
    return refiner.eval().requires_grad_(False), {
        "initial_state_hash": initial_hash,
        "final_state_hash": _state_hash(refiner),
        "decoder_hash_before": decoder_hash_before,
        "decoder_hash_after": decoder_hash_after,
        "decoder_unchanged": True,
        "training_order_hash": order_digest.hexdigest(),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _batch_metadata(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    def item(name: str, fallback: Any) -> Any:
        if name not in batch:
            return fallback
        value = batch[name]
        if isinstance(value, Tensor):
            return value[index].item()
        return value[index]

    return {
        "sample_id": int(item("sample_id", index)),
        "family": str(item("family", "unknown")),
        "model_id": str(item("model_id", item("sample_id", index))),
        "normals_estimated": bool(item("normals_estimated", False)),
    }


def _serialized_coordinates(
    coordinates: Tensor,
    *,
    config: StabilityExperimentConfig,
    mode: int | str = MODE_FIXED,
    normalization_centers: Tensor | None = None,
    normalization_scales: Tensor | None = None,
) -> tuple[Tensor, list[ConstellationPacket], bool, bool]:
    if (normalization_centers is None) != (normalization_scales is None):
        raise ValueError("normalization centers and scales must be supplied together")
    if normalization_centers is not None and len(normalization_centers) < len(
        coordinates
    ):
        raise ValueError("normalization batch is smaller than the coordinate batch")
    decoded = []
    packets = []
    exact = True
    lattice_exact = True
    for index, row in enumerate(coordinates.detach().cpu().numpy()):
        center = (
            None
            if normalization_centers is None
            else normalization_centers[index].detach().cpu().numpy()
        )
        scale = (
            None
            if normalization_scales is None
            else float(normalization_scales[index].item())
        )
        stream = encode_constellation(
            row,
            bits=config.coordinate_bits,
            mode=mode,
            output_points=config.num_points,
            normalization_center=center,
            normalization_scale=scale,
        )
        packet = decode_constellation(stream)
        levels = (1 << packet.bits) - 1
        lattice_values = (packet.normalized_coordinates + 1.0) * 0.5 * levels
        lattice_exact = lattice_exact and bool(
            np.all(np.abs(lattice_values - np.rint(lattice_values)) <= 1e-9)
        )
        exact = (
            exact
            and encode_constellation(
                packet.normalized_coordinates,
                bits=packet.bits,
                mode=packet.mode,
                output_points=packet.output_points,
                normalization_center=packet.normalization_center,
                normalization_scale=packet.normalization_scale,
            )
            == stream
        )
        decoded.append(torch.from_numpy(packet.normalized_coordinates).float())
        packets.append(packet)
    return (
        torch.stack(decoded).to(coordinates.device),
        packets,
        exact,
        lattice_exact,
    )


def _entropy_rate_fields(
    packet: ConstellationPacket, *, num_points: int
) -> dict[str, float | int]:
    """Return exact optional-stream rates without changing the declared packet."""

    entropy_stream = encode_constellation(
        packet.normalized_coordinates,
        bits=packet.bits,
        mode=MODE_ENTROPY,
        output_points=packet.output_points,
        normalization_center=packet.normalization_center,
        normalization_scale=packet.normalization_scale,
    )
    entropy_packet = decode_constellation(entropy_stream)
    if not np.array_equal(packet.coordinates, entropy_packet.coordinates):
        raise RuntimeError("entropy stream changed the fixed-stream lattice")
    return {
        "entropy_stream_bytes": len(entropy_stream),
        "entropy_bpp": 8.0 * len(entropy_stream) / num_points,
        "entropy_bound_bytes": entropy_bound_bytes(
            packet.normalized_coordinates, bits=packet.bits
        )
        + packet.normalization_bytes,
    }


def _evaluate_pair(
    config: StabilityExperimentConfig,
    datasets: DatasetMap,
    *,
    decoder: nn.Module,
    refiner: nn.Module,
    arm: str,
    decoder_seed: int,
    refiner_seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_adam_probe = refiner_seed == config.refiner_seeds[0]
    for split in EVALUATION_SPLITS:
        probed = 0
        for batch in _loader(datasets[split], config=config, shuffle=False):
            source = _source(batch, device)
            fresh = _fresh_target(batch, source)
            normalization_centers = batch.get("normalization_center")
            normalization_scales = batch.get("normalization_scale")
            fps = _fps(source, config.constellation_size, config.coordinate_bits)
            refiner_coordinates = refiner(
                source,
                config.constellation_size,
                decoder=decoder,
                target=source,
                num_output_points=config.num_points,
            )
            methods: dict[str, tuple[Tensor, bool]] = {
                "fps": (fps, False),
                "refiner": (refiner_coordinates, False),
            }
            remaining = (
                config.adam_probe_clouds_per_split - probed if run_adam_probe else 0
            )
            probe_count = min(len(source), max(remaining, 0))
            if config.adam_probe_evaluations and probe_count:
                probe_source = source[:probe_count]
                # The preregistered per-cloud optimization probe starts from
                # the same fixed FPS message in every factorial cell.
                initial = fps[:probe_count]

                def score(
                    candidate: Tensor, source_target: Tensor = probe_source
                ) -> Tensor:
                    reconstruction = decoder(
                        candidate, num_output_points=config.num_points
                    )
                    return _per_cloud_chamfer(
                        reconstruction,
                        source_target,
                        chunk_size=config.distance_chunk_size,
                    )

                searched = adam_ste_search(
                    score,
                    initial,
                    bits=config.coordinate_bits,
                    decoder_evaluation_budget=config.adam_probe_evaluations,
                    learning_rate=config.adam_probe_learning_rate,
                )
                methods["adam_probe"] = (searched.coordinates, True)
                probed += probe_count

            for method, (coordinates, source_only_probe) in methods.items():
                mode = "fps" if method == "fps" else "free"
                decoded, packets, exact, lattice_exact = _serialized_coordinates(
                    coordinates,
                    config=config,
                    mode=mode,
                    normalization_centers=normalization_centers,
                    normalization_scales=normalization_scales,
                )
                with torch.no_grad():
                    reconstruction = decoder(
                        decoded, num_output_points=config.num_points
                    )
                    source_losses = _per_cloud_chamfer(
                        reconstruction,
                        source[: len(decoded)],
                        chunk_size=config.distance_chunk_size,
                    )
                    fresh_losses = _per_cloud_chamfer(
                        reconstruction,
                        fresh[: len(decoded)],
                        chunk_size=config.distance_chunk_size,
                )
                for index in range(len(decoded)):
                    metadata = _batch_metadata(batch, index)
                    packet = packets[index]
                    reconstruction_points = reconstruction[index].detach().cpu().numpy()
                    source_points = source[index].detach().cpu().numpy()
                    fresh_points = fresh[index].detach().cpu().numpy()
                    source_tail = point_set_error_metrics(
                        reconstruction_points,
                        source_points,
                        chunk_size=config.distance_chunk_size,
                    )
                    fresh_tail = point_set_error_metrics(
                        reconstruction_points,
                        fresh_points,
                        chunk_size=config.distance_chunk_size,
                    )
                    continuous_surface: dict[str, float] = {}
                    if config.compute_mesh_metrics:
                        dataset = datasets[split]
                        if not isinstance(dataset, MeshSurfaceDataset):
                            raise ValueError(
                                "compute_mesh_metrics requires a mesh manifest dataset"
                            )
                        continuous_surface = mesh_surface_metrics(
                            reconstruction_points,
                            dataset.mesh(metadata["sample_id"]),
                            point_chunk_size=config.mesh_metric_point_chunk_size,
                            triangle_chunk_size=(
                                config.mesh_metric_triangle_chunk_size
                            ),
                            normal_neighbors=config.mesh_normal_neighbors,
                        )
                    rows.append(
                        {
                            "arm": arm,
                            "decoder_seed": decoder_seed,
                            "refiner_seed": refiner_seed,
                            "pair_id": (
                                f"{arm}:decoder_{decoder_seed}:refiner_{refiner_seed}"
                            ),
                            "split": split,
                            **metadata,
                            "method": method,
                            "constellation_size": config.constellation_size,
                            "coordinate_bits": config.coordinate_bits,
                            "header_bytes": packet.header_bytes,
                            "payload_bytes": packet.payload_bytes,
                            "normalization_bytes": packet.normalization_bytes,
                            "payload_bpp": 8.0
                            * packet.payload_bytes
                            / config.num_points,
                            "stream_bytes": packet.stream_bytes,
                            "actual_stream_bpp": 8.0
                            * packet.stream_bytes
                            / config.num_points,
                            **_entropy_rate_fields(
                                packet, num_points=config.num_points
                            ),
                            "chamfer_mse": float(source_losses[index].item()),
                            "fresh_chamfer_mse": float(fresh_losses[index].item()),
                            "p90_euclidean": source_tail["p90_euclidean"],
                            "p99_euclidean": source_tail["p99_euclidean"],
                            "hausdorff": source_tail["hausdorff"],
                            "fresh_p90_euclidean": fresh_tail["p90_euclidean"],
                            "fresh_p99_euclidean": fresh_tail["p99_euclidean"],
                            "fresh_hausdorff": fresh_tail["hausdorff"],
                            **continuous_surface,
                            "source_only_optimization": source_only_probe,
                            "serialized_round_trip_exact": exact,
                            "coordinates_on_exact_lattice": lattice_exact,
                            "adam_decoder_evaluations": (
                                config.adam_probe_evaluations
                                if source_only_probe
                                else None
                            ),
                        }
                    )
    return rows


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["arm"],
            row["decoder_seed"],
            row["refiner_seed"],
            row["split"],
            row["method"],
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        arm, decoder_seed, refiner_seed, split, method = key
        mse = float(np.mean([row["chamfer_mse"] for row in group]))
        fresh_mse = float(np.mean([row["fresh_chamfer_mse"] for row in group]))
        result.append(
            {
                "arm": arm,
                "decoder_seed": decoder_seed,
                "refiner_seed": refiner_seed,
                "pair_id": f"{arm}:decoder_{decoder_seed}:refiner_{refiner_seed}",
                "split": split,
                "method": method,
                "clouds": len(group),
                "header_bytes": float(np.mean([row["header_bytes"] for row in group])),
                "payload_bytes": float(
                    np.mean([row["payload_bytes"] for row in group])
                ),
                "normalization_bytes": float(
                    np.mean([row.get("normalization_bytes", 0) for row in group])
                ),
                "stream_bytes": float(np.mean([row["stream_bytes"] for row in group])),
                "actual_stream_bpp": float(
                    np.mean([row["actual_stream_bpp"] for row in group])
                ),
                "chamfer_mse": mse,
                "chamfer_rmse": math.sqrt(mse),
                "fresh_chamfer_mse": fresh_mse,
                "fresh_chamfer_rmse": math.sqrt(fresh_mse),
                **{
                    field: float(np.mean([item[field] for item in group]))
                    for field in (
                        "p90_euclidean",
                        "p99_euclidean",
                        "hausdorff",
                        "fresh_p90_euclidean",
                        "fresh_p99_euclidean",
                        "fresh_hausdorff",
                        "surface_rmse",
                        "normal_consistency",
                    )
                    if all(field in item for item in group)
                },
            }
        )
    fps = {
        (row["arm"], row["decoder_seed"], row["refiner_seed"], row["split"]): row
        for row in result
        if row["method"] == "fps"
    }
    for row in result:
        if row["method"] != "fps":
            baseline = fps[
                (
                    row["arm"],
                    row["decoder_seed"],
                    row["refiner_seed"],
                    row["split"],
                )
            ]
            if row["method"] == "adam_probe":
                method_group = groups[
                    (
                        row["arm"],
                        row["decoder_seed"],
                        row["refiner_seed"],
                        row["split"],
                        row["method"],
                    )
                ]
                cloud_keys = {
                    (item["family"], item["model_id"]) for item in method_group
                }
                fps_group = [
                    item
                    for item in groups[
                        (
                            row["arm"],
                            row["decoder_seed"],
                            row["refiner_seed"],
                            row["split"],
                            "fps",
                        )
                    ]
                    if (item["family"], item["model_id"]) in cloud_keys
                ]
                if len(fps_group) != len(method_group):
                    raise RuntimeError("Adam and matched FPS diagnostic subsets differ")
                baseline = {
                    **baseline,
                    "chamfer_rmse": math.sqrt(
                        float(np.mean([item["chamfer_mse"] for item in fps_group]))
                    ),
                    "fresh_chamfer_rmse": math.sqrt(
                        float(
                            np.mean([item["fresh_chamfer_mse"] for item in fps_group])
                        )
                    ),
                }
                row["comparison_subset"] = "same_clouds_as_adam_probe"
            row["relative_rmse_improvement_over_fps_percent"] = (
                100.0
                * (baseline["chamfer_rmse"] - row["chamfer_rmse"])
                / max(baseline["chamfer_rmse"], 1e-12)
            )
            row["fresh_relative_rmse_improvement_over_fps_percent"] = (
                100.0
                * (baseline["fresh_chamfer_rmse"] - row["fresh_chamfer_rmse"])
                / max(baseline["fresh_chamfer_rmse"], 1e-12)
            )
    return result


def _amortize_stability_summaries(
    summaries: list[dict[str, Any]],
    decoder_records: list[dict[str, Any]],
    *,
    input_points: int,
) -> list[dict[str, Any]]:
    model_bytes = {
        (record["decoder_seed"], arm): {
            precision: file_record["bytes"]
            for precision, file_record in arm_record["decoder_state_dicts"].items()
        }
        for record in decoder_records
        for arm, arm_record in record["arms"].items()
    }
    return [
        {
            **row,
            **model_amortization(
                row["stream_bytes"],
                input_points,
                model_bytes[(row["decoder_seed"], row["arm"])],
            ),
        }
        for row in summaries
    ]


def _two_way_components(
    matrix: np.ndarray,
    *,
    decoder_seeds: Iterable[int],
    refiner_seeds: Iterable[int],
) -> dict[str, Any]:
    """Method-of-moments components for a balanced crossed cell-mean design."""

    decoder_seeds = tuple(decoder_seeds)
    refiner_seeds = tuple(refiner_seeds)
    decoders, refiners = matrix.shape
    grand = float(matrix.mean())
    decoder_means = matrix.mean(axis=1)
    refiner_means = matrix.mean(axis=0)
    residual = matrix - decoder_means[:, None] - refiner_means[None, :] + grand
    ms_decoder = refiners * float(((decoder_means - grand) ** 2).sum()) / (decoders - 1)
    ms_refiner = decoders * float(((refiner_means - grand) ** 2).sum()) / (refiners - 1)
    ms_interaction = float((residual**2).sum()) / ((decoders - 1) * (refiners - 1))
    decoder_unconstrained = (ms_decoder - ms_interaction) / refiners
    refiner_unconstrained = (ms_refiner - ms_interaction) / decoders
    decoder_variance = max(decoder_unconstrained, 0.0)
    refiner_variance = max(refiner_unconstrained, 0.0)
    interaction_variance = ms_interaction
    total = decoder_variance + refiner_variance + interaction_variance
    return {
        "estimator": "balanced_two_way_random_effects_method_of_moments_on_cell_means",
        "matrix_decoder_rows_refiner_columns": matrix.tolist(),
        "decoder_seeds": list(decoder_seeds),
        "refiner_seeds": list(refiner_seeds),
        "grand_mean": grand,
        "mean_squares": {
            "decoder": ms_decoder,
            "refiner": ms_refiner,
            "decoder_refiner_interaction": ms_interaction,
        },
        "unconstrained_variance": {
            "decoder": decoder_unconstrained,
            "refiner": refiner_unconstrained,
            "decoder_refiner_interaction": interaction_variance,
        },
        "nonnegative_constrained_variance": {
            "decoder": decoder_variance,
            "refiner": refiner_variance,
            "decoder_refiner_interaction": interaction_variance,
        },
        "variance": {
            "decoder": decoder_variance,
            "refiner": refiner_variance,
            "decoder_refiner_interaction": interaction_variance,
        },
        "variance_fraction": {
            "decoder": decoder_variance / total if total else 0.0,
            "refiner": refiner_variance / total if total else 0.0,
            "decoder_refiner_interaction": (
                interaction_variance / total if total else 0.0
            ),
        },
    }


def _variance_components(
    summaries: list[dict[str, Any]], config: StabilityExperimentConfig
) -> list[dict[str, Any]]:
    indexed = {
        (
            row["arm"],
            row["split"],
            row["method"],
            row["decoder_seed"],
            row["refiner_seed"],
        ): row
        for row in summaries
    }
    results = []
    component_index = 0
    for arm in ARMS:
        for split in EVALUATION_SPLITS:
            for metric in ("chamfer_rmse", "fresh_chamfer_rmse"):
                rmse_matrix = np.asarray(
                    [
                        [
                            indexed[
                                (
                                    arm,
                                    split,
                                    "refiner",
                                    decoder_seed,
                                    refiner_seed,
                                )
                            ][metric]
                            for refiner_seed in config.refiner_seeds
                        ]
                        for decoder_seed in config.decoder_seeds
                    ],
                    dtype=np.float64,
                )
                matrix = np.log(rmse_matrix.clip(min=1e-12))
                components = _two_way_components(
                    matrix,
                    decoder_seeds=config.decoder_seeds,
                    refiner_seeds=config.refiner_seeds,
                )
                variance = components["variance"]
                denominator = (
                    variance["refiner"] + variance["decoder_refiner_interaction"]
                )
                point_ratio = variance["decoder"] / max(denominator, 1e-18)
                rng = np.random.default_rng(
                    config.bootstrap_seed + 50_000 + component_index
                )
                ratios = np.empty(config.bootstrap_samples, dtype=np.float64)
                for bootstrap_index in range(config.bootstrap_samples):
                    decoder_draw = rng.integers(
                        0, matrix.shape[0], size=matrix.shape[0]
                    )
                    refiner_draw = rng.integers(
                        0, matrix.shape[1], size=matrix.shape[1]
                    )
                    draw = matrix[decoder_draw[:, None], refiner_draw[None, :]]
                    draw_components = _two_way_components(
                        draw,
                        decoder_seeds=config.decoder_seeds,
                        refiner_seeds=config.refiner_seeds,
                    )["variance"]
                    draw_denominator = (
                        draw_components["refiner"]
                        + draw_components["decoder_refiner_interaction"]
                    )
                    ratios[bootstrap_index] = draw_components["decoder"] / max(
                        draw_denominator, 1e-18
                    )
                alpha = (1.0 - config.confidence_level) / 2.0
                lower, upper = np.quantile(ratios, (alpha, 1.0 - alpha))
                diagnosis = (
                    "decoder_dominant"
                    if lower > 1.0
                    else "refiner_or_interaction_dominant"
                    if upper < 1.0
                    else "inconclusive"
                )
                results.append(
                    {
                        "arm": arm,
                        "split": split,
                        "method": "refiner",
                        "metric": metric,
                        "analysis_scale": "natural_log_of_per_pair_aggregate_rmse",
                        "rmse_matrix_decoder_rows_refiner_columns": (
                            rmse_matrix.tolist()
                        ),
                        **components,
                        "decoder_vs_refiner_interaction_ratio": {
                            "point": point_ratio,
                            "confidence_interval_lower": float(lower),
                            "confidence_interval_upper": float(upper),
                            "bootstrap_samples": config.bootstrap_samples,
                            "diagnosis": diagnosis,
                        },
                    }
                )
                component_index += 1
    return results


def _rank_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or float(first.std()) == 0.0 or float(second.std()) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _calibration_test_associations(
    summaries: list[dict[str, Any]],
    decoder_records: list[dict[str, Any]],
    config: StabilityExperimentConfig,
) -> list[dict[str, Any]]:
    """Associate source-visible FPS calibration with decoder-only FPS test quality."""

    selected = {
        (record["decoder_seed"], arm): record["arms"][arm]["selection"]
        for record in decoder_records
        for arm in ARMS
    }
    associations = []
    for arm in ARMS:
        for split in EVALUATION_SPLITS:
            for metric in ("chamfer_rmse", "fresh_chamfer_rmse"):
                calibration = []
                evaluation = []
                points = []
                for decoder_seed in config.decoder_seeds:
                    cell_values = [
                        row[metric]
                        for row in summaries
                        if row["arm"] == arm
                        and row["split"] == split
                        and row["method"] == "fps"
                        and row["decoder_seed"] == decoder_seed
                    ]
                    if len(cell_values) != len(config.refiner_seeds):
                        raise RuntimeError(
                            "calibration association has incomplete cells"
                        )
                    calibration_score = selected[(decoder_seed, arm)][
                        "calibration_chamfer_rmse"
                    ]
                    evaluation_score = float(np.mean(cell_values))
                    calibration.append(calibration_score)
                    evaluation.append(evaluation_score)
                    points.append(
                        {
                            "decoder_seed": decoder_seed,
                            "calibration_chamfer_rmse": calibration_score,
                            "decoder_fps_test_rmse": evaluation_score,
                        }
                    )
                calibration_array = np.asarray(calibration, dtype=np.float64)
                evaluation_array = np.asarray(evaluation, dtype=np.float64)
                associations.append(
                    {
                        "arm": arm,
                        "split": split,
                        "method": "serialized_fps_decoder_only",
                        "test_metric": metric,
                        "independent_units": "decoder_seeds",
                        "n": len(config.decoder_seeds),
                        "spearman_primary": _correlation(
                            _rank_average_ties(calibration_array),
                            _rank_average_ties(evaluation_array),
                        ),
                        "pearson_secondary": _correlation(
                            calibration_array, evaluation_array
                        ),
                        "points": points,
                    }
                )
    return associations


def _stability_gates(
    summaries: list[dict[str, Any]], config: StabilityExperimentConfig
) -> dict[str, Any]:
    """Report bad-seed-tail gates over independent decoder marginals."""

    results = []
    for split_index, split in enumerate(EVALUATION_SPLITS):
        for metric in ("chamfer_rmse", "fresh_chamfer_rmse"):
            mse_metric = metric.replace("rmse", "mse")
            values: dict[tuple[str, str], np.ndarray] = {}
            for arm in ARMS:
                for method in ("fps", "refiner"):
                    decoder_marginals = []
                    for decoder_seed in config.decoder_seeds:
                        cells = [
                            row[mse_metric]
                            for row in summaries
                            if row["arm"] == arm
                            and row["split"] == split
                            and row["method"] == method
                            and row["decoder_seed"] == decoder_seed
                        ]
                        if len(cells) != len(config.refiner_seeds):
                            raise RuntimeError("stability gate has incomplete cells")
                        decoder_marginals.append(math.sqrt(float(np.mean(cells))))
                    values[(arm, method)] = np.asarray(decoder_marginals)
            baseline = values[("baseline", "refiner")]
            stabilized = values[("stabilized", "refiner")]
            stabilized_fps = values[("stabilized", "fps")]
            baseline_q90 = float(np.quantile(baseline, 0.9))
            stabilized_q90 = float(np.quantile(stabilized, 0.9))
            q90_reduction = (
                100.0 * (baseline_q90 - stabilized_q90) / max(baseline_q90, 1e-12)
            )
            median_change = (
                100.0
                * (float(np.median(baseline)) - float(np.median(stabilized)))
                / max(float(np.median(baseline)), 1e-12)
            )
            decoder_fps_gains = (
                100.0
                * (stabilized_fps - stabilized)
                / np.clip(stabilized_fps, 1e-12, None)
            )
            rng = np.random.default_rng(
                config.bootstrap_seed + 10_000 * split_index + len(results)
            )
            indices = rng.integers(
                0,
                len(config.decoder_seeds),
                size=(config.bootstrap_samples, len(config.decoder_seeds)),
            )
            baseline_draws = np.quantile(baseline[indices], 0.9, axis=1)
            stabilized_draws = np.quantile(stabilized[indices], 0.9, axis=1)
            improvements = (
                100.0
                * (baseline_draws - stabilized_draws)
                / np.clip(baseline_draws, 1e-12, None)
            )
            alpha = (1.0 - config.confidence_level) / 2.0
            lower, upper = np.quantile(improvements, (alpha, 1.0 - alpha))
            validation_material = (
                q90_reduction >= 5.0
                and float(lower) > 0.0
                and median_change >= -1.0
                and float(np.median(decoder_fps_gains)) >= 10.0
            )
            results.append(
                {
                    "split": split,
                    "metric": metric,
                    "decoder_marginals": len(stabilized),
                    "refiner_cells_per_decoder": len(config.refiner_seeds),
                    "uses_all_predeclared_cells": True,
                    "baseline_q90": baseline_q90,
                    "stabilized_q90": stabilized_q90,
                    "q90_relative_rmse_reduction_percent": q90_reduction,
                    "q90_reduction_ci_lower_percent": float(lower),
                    "q90_reduction_ci_upper_percent": float(upper),
                    "baseline_max": float(baseline.max()),
                    "stabilized_max": float(stabilized.max()),
                    "median_relative_rmse_reduction_percent": median_change,
                    "median_decoder_level_improvement_over_fps_percent": float(
                        np.median(decoder_fps_gains)
                    ),
                    "stabilized_decoders_better_than_paired_baseline": int(
                        np.count_nonzero(stabilized < baseline)
                    ),
                    "stabilized_decoders_better_than_matched_fps": int(
                        np.count_nonzero(stabilized < stabilized_fps)
                    ),
                    "validation_material_thresholds_pass": validation_material,
                }
            )
    primary = next(
        row
        for row in results
        if row["split"] == "validation" and row["metric"] == "chamfer_rmse"
    )
    ood_primary = next(
        row
        for row in results
        if row["split"] == "ood" and row["metric"] == "chamfer_rmse"
    )
    return {
        "definition": (
            "Q90 over decoder-marginal RMSE; validation requires >=5% reduction "
            "with positive bootstrap lower bound, median no worse than -1%, and "
            ">=10% median FPS gain; OOD Q90 and FPS gain must remain positive"
        ),
        "primary": primary,
        "ood_primary": ood_primary,
        "overall_stability_gate_passes": bool(
            primary["validation_material_thresholds_pass"]
            and ood_primary["q90_relative_rmse_reduction_percent"] > 0.0
            and ood_primary["median_decoder_level_improvement_over_fps_percent"] >= 10.0
        ),
        "all_metrics": results,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _hierarchical_cloud_draw(
    families: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    categories = np.asarray(sorted(set(families.tolist())), dtype=object)
    drawn_categories = categories[
        rng.integers(0, len(categories), size=len(categories))
    ]
    drawn_clouds = []
    for category in drawn_categories:
        members = np.flatnonzero(families == category)
        drawn_clouds.extend(
            members[rng.integers(0, len(members), size=len(members))].tolist()
        )
    return np.asarray(drawn_clouds, dtype=np.int64)


def _independent_feature_bootstrap(
    coordinate: np.ndarray,
    feature: np.ndarray,
    families: np.ndarray,
    *,
    config: StabilityExperimentConfig,
    seed_offset: int,
) -> dict[str, Any]:
    """Compare independent model factors while pairing category/cloud draws."""

    if coordinate.ndim != 3 or feature.ndim != 2:
        raise ValueError("coordinate and feature losses have invalid dimensions")
    if coordinate.shape[2] != feature.shape[1] or len(families) != feature.shape[1]:
        raise ValueError("coordinate/feature cloud axes do not align")
    coordinate_rmse = math.sqrt(float(coordinate.mean()))
    feature_rmse = math.sqrt(float(feature.mean()))
    point = 100.0 * (feature_rmse - coordinate_rmse) / max(feature_rmse, 1e-12)
    rng = np.random.default_rng(config.bootstrap_seed + seed_offset)
    draws = np.empty(config.bootstrap_samples, dtype=np.float64)
    for index in range(config.bootstrap_samples):
        decoder_draw = rng.integers(0, coordinate.shape[0], size=coordinate.shape[0])
        refiner_draw = rng.integers(0, coordinate.shape[1], size=coordinate.shape[1])
        feature_draw = rng.integers(0, feature.shape[0], size=feature.shape[0])
        cloud_draw = _hierarchical_cloud_draw(families, rng)
        coordinate_mse = float(
            coordinate[
                decoder_draw[:, None, None],
                refiner_draw[None, :, None],
                cloud_draw[None, None, :],
            ].mean()
        )
        feature_mse = float(feature[feature_draw[:, None], cloud_draw[None, :]].mean())
        feature_draw_rmse = math.sqrt(feature_mse)
        draws[index] = (
            100.0
            * (feature_draw_rmse - math.sqrt(coordinate_mse))
            / max(feature_draw_rmse, 1e-12)
        )
    alpha = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(draws, (alpha, 1.0 - alpha))
    return {
        "coordinate_decoder_seeds": coordinate.shape[0],
        "coordinate_refiner_seeds": coordinate.shape[1],
        "feature_codec_seeds": feature.shape[0],
        "categories": len(set(families.tolist())),
        "clouds": len(families),
        "bootstrap_samples": config.bootstrap_samples,
        "coordinate_rmse": coordinate_rmse,
        "feature_codec_rmse": feature_rmse,
        "coordinate_relative_rmse_improvement_percent": point,
        "confidence_interval_lower_percent": float(lower),
        "confidence_interval_upper_percent": float(upper),
    }


def _feature_reference_comparison(
    rows: list[dict[str, Any]], config: StabilityExperimentConfig
) -> dict[str, Any]:
    """Compare all stabilized cells with independent frozen feature-codec seeds."""

    if config.reference_feature_dir is None:
        return {"status": "not_configured", "learned_codec_gate_passes": None}
    root = Path(config.reference_feature_dir)
    metrics_path = root / "multiseed_metrics.json"
    seed_paths = sorted(
        root.glob("seed_*/per_cloud.jsonl"),
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
    )
    if not metrics_path.is_file() or not seed_paths:
        return {
            "status": "configured_reference_missing",
            "reference_dir": str(root),
            "learned_codec_gate_passes": None,
        }
    reference_metrics = json.loads(metrics_path.read_text())
    reference_config = reference_metrics["config"]
    expected_bytes = expected_stream_bytes(
        config.constellation_size,
        config.coordinate_bits,
        normalization=config.dataset_kind == "mesh_manifest",
    )
    protocol_checks = {
        "data_seed_matches": reference_config["data_seed"] == config.data_seed,
        "num_points_matches": reference_config["num_points"] == config.num_points,
        "primary_constellation_size_matches": (
            reference_config["primary_constellation_size"] == config.constellation_size
        ),
        "coordinate_bits_matches": (
            reference_config["coordinate_bits"] == config.coordinate_bits
        ),
    }
    if not all(protocol_checks.values()):
        raise ValueError("feature reference protocol differs from Experiment 019")

    feature_seeds = tuple(
        int(path.parent.name.removeprefix("seed_")) for path in seed_paths
    )
    feature_rows = {
        seed: _read_jsonl(path)
        for seed, path in zip(feature_seeds, seed_paths, strict=True)
    }
    comparisons = []
    split_names = {"validation": "validation", "ood": "category_ood"}
    for split_index, (split, feature_split) in enumerate(split_names.items()):
        coordinate_rows = [
            row
            for row in rows
            if row["arm"] == "stabilized"
            and row["split"] == split
            and row["method"] == "refiner"
        ]
        first_pair = [
            row
            for row in coordinate_rows
            if row["decoder_seed"] == config.decoder_seeds[0]
            and row["refiner_seed"] == config.refiner_seeds[0]
        ]
        cloud_keys = [(row["family"], row["model_id"]) for row in first_pair]
        if len(cloud_keys) != len(set(cloud_keys)):
            raise RuntimeError("coordinate evaluation contains duplicate cloud keys")
        families = np.asarray([key[0] for key in cloud_keys], dtype=object)
        coordinate_index = {
            (
                row["decoder_seed"],
                row["refiner_seed"],
                row["family"],
                row["model_id"],
            ): row
            for row in coordinate_rows
        }
        filtered_feature: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
        for seed, seed_rows in feature_rows.items():
            selected = [
                row
                for row in seed_rows
                if row["split"] == feature_split
                and row["method"] == "feature_latent"
                and row["constellation_size"] == config.constellation_size
            ]
            if any(
                row["stream_bytes"]
                + (
                    0
                    if "normalization_bytes" in row
                    else (
                        expected_stream_bytes(
                            1, config.coordinate_bits, normalization=True
                        )
                        - expected_stream_bytes(1, config.coordinate_bits)
                        if config.dataset_kind == "mesh_manifest"
                        else 0
                    )
                )
                != expected_bytes
                for row in selected
            ):
                raise ValueError("feature reference stream size is not rate matched")
            indexed = {(row["family"], row["model_id"]): row for row in selected}
            if list(indexed) != cloud_keys and set(indexed) != set(cloud_keys):
                raise ValueError("feature reference cloud identities do not align")
            filtered_feature[seed] = indexed

        for metric_index, metric in enumerate(("chamfer_mse", "fresh_chamfer_mse")):
            coordinate = np.asarray(
                [
                    [
                        [
                            coordinate_index[(decoder_seed, refiner_seed, *key)][metric]
                            for key in cloud_keys
                        ]
                        for refiner_seed in config.refiner_seeds
                    ]
                    for decoder_seed in config.decoder_seeds
                ],
                dtype=np.float64,
            )
            feature = np.asarray(
                [
                    [filtered_feature[seed][key][metric] for key in cloud_keys]
                    for seed in feature_seeds
                ],
                dtype=np.float64,
            )
            comparisons.append(
                {
                    "split": split,
                    "metric": metric,
                    **_independent_feature_bootstrap(
                        coordinate,
                        feature,
                        families,
                        config=config,
                        seed_offset=1000 * split_index + metric_index,
                    ),
                }
            )
    primary = next(
        row
        for row in comparisons
        if row["split"] == "validation" and row["metric"] == "chamfer_mse"
    )
    return {
        "status": "complete",
        "reference_dir": str(root),
        "feature_seeds": list(feature_seeds),
        "protocol_checks": protocol_checks,
        "resampling": (
            "independent coordinate-decoder, coordinate-refiner, and feature-seed "
            "factors with paired hierarchical category/cloud draws"
        ),
        "comparisons": comparisons,
        "primary": primary,
        "learned_codec_gate_passes": (
            primary["confidence_interval_lower_percent"] > 0.0
        ),
        "d1_d2_status": "not_run_in_factorial; requires selected official-metric pass",
    }


def _arm_comparisons(
    summaries: list[dict[str, Any]], config: StabilityExperimentConfig
) -> list[dict[str, Any]]:
    indexed = {
        (
            row["arm"],
            row["decoder_seed"],
            row["refiner_seed"],
            row["split"],
            row["method"],
        ): row
        for row in summaries
    }
    comparisons = []
    for decoder_seed in config.decoder_seeds:
        for refiner_seed in config.refiner_seeds:
            for split in EVALUATION_SPLITS:
                for method in ("fps", "refiner"):
                    baseline = indexed[
                        ("baseline", decoder_seed, refiner_seed, split, method)
                    ]
                    stabilized = indexed[
                        ("stabilized", decoder_seed, refiner_seed, split, method)
                    ]
                    comparisons.append(
                        {
                            "decoder_seed": decoder_seed,
                            "refiner_seed": refiner_seed,
                            "split": split,
                            "method": method,
                            "baseline_chamfer_rmse": baseline["chamfer_rmse"],
                            "stabilized_chamfer_rmse": stabilized["chamfer_rmse"],
                            "stabilized_relative_rmse_improvement_percent": 100.0
                            * (baseline["chamfer_rmse"] - stabilized["chamfer_rmse"])
                            / max(baseline["chamfer_rmse"], 1e-12),
                        }
                    )
    return comparisons


def run_stability_experiment(
    config: StabilityExperimentConfig, *, device_name: str = "auto"
) -> dict[str, Any]:
    """Execute the complete decoder-arm by decoder/refiner-seed factorial."""

    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _datasets(config)
    data_protocol = _data_protocol(config, datasets)
    if not data_protocol["all_partitions_pairwise_disjoint"]:
        raise RuntimeError("training/calibration/evaluation memberships overlap")

    started = time.perf_counter()
    decoder_records = []
    pair_records = []
    all_rows: list[dict[str, Any]] = []
    for decoder_seed in config.decoder_seeds:
        arm_states, decoder_record = _train_decoders(
            config,
            datasets,
            decoder_seed=decoder_seed,
            device=device,
            output_dir=output_dir,
        )
        decoder_records.append(decoder_record)
        for arm in ARMS:
            for refiner_seed in config.refiner_seeds:
                decoder = _decoder(config, device)
                decoder.load_state_dict(arm_states[arm])
                decoder.eval().requires_grad_(False)
                refiner, training = _train_refiner(
                    config,
                    datasets,
                    decoder=decoder,
                    refiner_seed=refiner_seed,
                    device=device,
                )
                pair_dir = (
                    output_dir
                    / "pairs"
                    / arm
                    / f"decoder_{decoder_seed}_refiner_{refiner_seed}"
                )
                pair_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": refiner.state_dict(),
                        "arm": arm,
                        "decoder_seed": decoder_seed,
                        "refiner_seed": refiner_seed,
                        "decoder_state_hash": training["decoder_hash_before"],
                    },
                    pair_dir / "refiner.pt",
                )
                rows = _evaluate_pair(
                    config,
                    datasets,
                    decoder=decoder,
                    refiner=refiner,
                    arm=arm,
                    decoder_seed=decoder_seed,
                    refiner_seed=refiner_seed,
                    device=device,
                )
                all_rows.extend(rows)
                pair_record = {
                    "pair_id": f"{arm}:decoder_{decoder_seed}:refiner_{refiner_seed}",
                    "arm": arm,
                    "decoder_seed": decoder_seed,
                    "refiner_seed": refiner_seed,
                    "calibration_selection": decoder_record["arms"][arm]["selection"],
                    "calibration_partition_sha256": data_protocol["partitions"][
                        "calibration"
                    ]["sha256"],
                    "evaluation_partition_sha256": {
                        split: data_protocol["partitions"][split]["sha256"]
                        for split in EVALUATION_SPLITS
                    },
                    "training": training,
                    "per_cloud_rows": len(rows),
                }
                pair_records.append(pair_record)
                (pair_dir / "pair_metrics.json").write_text(
                    json.dumps(pair_record, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {
                            "stage": "pair_complete",
                            "pair_id": pair_record["pair_id"],
                            "rows": len(rows),
                        }
                    ),
                    flush=True,
                )

    summaries = _amortize_stability_summaries(
        _summaries(all_rows), decoder_records, input_points=config.num_points
    )
    initial_hashes_matched = all(
        len(
            {
                pair["training"]["initial_state_hash"]
                for pair in pair_records
                if pair["refiner_seed"] == refiner_seed
            }
        )
        == 1
        for refiner_seed in config.refiner_seeds
    )
    orders_matched = all(
        len(
            {
                pair["training"]["training_order_hash"]
                for pair in pair_records
                if pair["refiner_seed"] == refiner_seed
            }
        )
        == 1
        for refiner_seed in config.refiner_seeds
    )
    result = {
        "experiment": "019_decoder_refiner_stability_decomposition",
        "config": asdict(config),
        "device": str(device),
        "scientific_contract": {
            "learned_message": "unordered quantized K x 3 coordinates only",
            "per_object_side_information": (
                "8-byte binary16 center/scale normalization"
                if config.dataset_kind == "mesh_manifest"
                else "none"
            ),
            "source_only_encoder_feedback": True,
            "actual_bitstream_round_trip": True,
            "decoder_frozen_during_refiner_and_adam": True,
            "calibration_excludes_validation_and_ood": True,
        },
        "data_protocol": data_protocol,
        "factorial": {
            "decoder_seeds": list(config.decoder_seeds),
            "refiner_seeds": list(config.refiner_seeds),
            "arms": list(ARMS),
            "cells": len(config.decoder_seeds) * len(config.refiner_seeds) * len(ARMS),
            "complete": len(pair_records)
            == len(config.decoder_seeds) * len(config.refiner_seeds) * len(ARMS),
        },
        "decoder_records": decoder_records,
        "pair_associations": pair_records,
        "summary": summaries,
        "variance_components": _variance_components(summaries, config),
        "calibration_test_associations": _calibration_test_associations(
            summaries, decoder_records, config
        ),
        "stability_gates": _stability_gates(summaries, config),
        "feature_reference": _feature_reference_comparison(all_rows, config),
        "arm_comparisons": _arm_comparisons(summaries, config),
        "contract_checks": {
            "all_decoder_hashes_frozen_within_pairs": all(
                pair["training"]["decoder_unchanged"] for pair in pair_records
            ),
            "all_stream_round_trips_exact": all(
                row["serialized_round_trip_exact"] for row in all_rows
            ),
            "all_coordinates_on_exact_lattice": all(
                row["coordinates_on_exact_lattice"] for row in all_rows
            ),
            "adam_probe_uses_source_only": all(
                row["source_only_optimization"]
                for row in all_rows
                if row["method"] == "adam_probe"
            ),
            "refiner_initial_hashes_matched_across_arms_and_decoders": (
                initial_hashes_matched
            ),
            "refiner_training_orders_matched_across_arms_and_decoders": (
                orders_matched
            ),
            "all_stream_sizes_match_declared_rate": all(
                row["stream_bytes"]
                == expected_stream_bytes(
                    config.constellation_size,
                    config.coordinate_bits,
                    normalization=config.dataset_kind == "mesh_manifest",
                )
                for row in all_rows
            ),
            "all_entropy_stream_rates_present": all(
                row["entropy_stream_bytes"] >= HEADER.size + 1
                and row["entropy_bpp"]
                == 8.0 * row["entropy_stream_bytes"] / config.num_points
                and row["entropy_bound_bytes"] <= row["entropy_stream_bytes"]
                for row in all_rows
            ),
            "all_header_payload_splits_exact": all(
                row["header_bytes"]
                + row["payload_bytes"]
                + row.get("normalization_bytes", 0)
                == row["stream_bytes"]
                for row in all_rows
            ),
        },
        "per_cloud": all_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (output_dir / "per_cloud.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "stability_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def reaggregate_stability_result(config: StabilityExperimentConfig) -> dict[str, Any]:
    """Recompute statistical summaries without retraining completed cells."""

    metrics_path = Path(config.output_dir) / "stability_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"completed stability metrics are absent: {metrics_path}"
        )
    result = json.loads(metrics_path.read_text())
    expected_config = json.loads(json.dumps(asdict(config)))
    if result.get("config") != expected_config:
        raise ValueError("completed result config does not match aggregate-only config")
    if not result.get("factorial", {}).get("complete"):
        raise ValueError("cannot aggregate an incomplete factorial result")
    rows = result["per_cloud"]
    summaries = _amortize_stability_summaries(
        _summaries(rows), result["decoder_records"], input_points=config.num_points
    )
    started = time.perf_counter()
    result.update(
        {
            "summary": summaries,
            "variance_components": _variance_components(summaries, config),
            "calibration_test_associations": _calibration_test_associations(
                summaries, result["decoder_records"], config
            ),
            "stability_gates": _stability_gates(summaries, config),
            "feature_reference": _feature_reference_comparison(rows, config),
            "arm_comparisons": _arm_comparisons(summaries, config),
            "aggregation_elapsed_seconds": time.perf_counter() - started,
        }
    )
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_019_stability_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    config = StabilityExperimentConfig.from_json(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    result = (
        reaggregate_stability_result(config)
        if args.aggregate_only
        else run_stability_experiment(config, device_name=args.device)
    )
    print(
        json.dumps(
            {
                "factorial": result["factorial"],
                "contract_checks": result["contract_checks"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "stability_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
