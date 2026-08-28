"""Experiment 033: source-only Adam/STE encoder-objective sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pointconstellation.bitstream import expected_stream_bytes
from pointconstellation.codecs import run_pc_error
from pointconstellation.data import file_sha256
from pointconstellation.headroom_experiment import _serialize_batch
from pointconstellation.losses import pairwise_squared
from pointconstellation.models.gradient_free import SearchResult, adam_ste_search
from pointconstellation.official_stability import OfficialStabilityConfig, _load_models
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.selection_baselines import SELECTION_METHODS
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _per_cloud_chamfer,
)
from pointconstellation.standardized_metrics import standardized_geometry_metrics
from pointconstellation.train import select_device, set_seed

OBJECTIVE_NAMES = ("chamfer", "point_to_plane", "feature_matching", "mixed")
SUPPORTED_SPLITS = ("validation", "ood")


class PointNetClassifier(nn.Module):
    """Small permutation-invariant classifier with exposed penultimate features."""

    def __init__(
        self,
        num_classes: int,
        *,
        feature_width: int = 64,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if feature_width < 4 or feature_dim < 4:
            raise ValueError("classifier feature dimensions must be at least four")
        self.num_classes = num_classes
        self.feature_width = feature_width
        self.feature_dim = feature_dim
        self.point_embedding = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.ReLU(),
            nn.Linear(feature_width, feature_width),
            nn.ReLU(),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(2 * feature_width, feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, points: Tensor) -> Tensor:
        """Return a fixed-dimensional permutation-invariant point-set feature."""

        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (batch, N, 3)")
        embedded = self.point_embedding(points)
        pooled = torch.cat((embedded.mean(dim=1), embedded.amax(dim=1)), dim=1)
        return self.feature_head(pooled)

    def forward(self, points: Tensor) -> Tensor:
        return self.classifier(self.extract_features(points))


@dataclass(frozen=True)
class ObjectiveContext:
    """All data an encode-time objective is permitted to inspect.

    Labels, fresh resamples, analytic normals, and evaluation targets are
    deliberately absent. Point-to-plane normals are estimated from
    ``source_points`` inside the objective builder.
    """

    decoder: nn.Module
    source_points: Tensor
    num_output_points: int
    distance_chunk_size: int
    normal_neighbors: int
    normal_chunk_size: int
    mixed_chamfer_weight: float
    feature_extractor: PointNetClassifier | None = None


ObjectiveScorer = Callable[[Tensor], Tensor]
ObjectiveBuilder = Callable[[ObjectiveContext], ObjectiveScorer]


def estimate_source_normals_pca(
    source_points: Tensor,
    *,
    neighbors: int,
    chunk_size: int,
) -> Tensor:
    """Estimate unoriented source normals with deterministic local PCA k-NN."""

    if source_points.ndim != 3 or source_points.shape[-1] != 3:
        raise ValueError("source_points must have shape (batch, N, 3)")
    if not 3 <= neighbors < source_points.shape[1]:
        raise ValueError("neighbors must be at least three and smaller than N")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    points = source_points.detach()
    normals = []
    with torch.no_grad():
        for start in range(0, points.shape[1], chunk_size):
            stop = min(start + chunk_size, points.shape[1])
            distances = pairwise_squared(points[:, start:stop], points)
            local_rows = torch.arange(stop - start, device=points.device)
            self_indices = torch.arange(start, stop, device=points.device)
            distances[:, local_rows, self_indices] = torch.inf
            indices = distances.topk(neighbors, dim=2, largest=False).indices
            expanded = points[:, None].expand(-1, stop - start, -1, -1)
            neighborhoods = expanded.gather(
                2, indices[:, :, :, None].expand(-1, -1, -1, 3)
            )
            centered = neighborhoods - neighborhoods.mean(dim=2, keepdim=True)
            covariance = centered.transpose(2, 3) @ centered / neighbors
            _, eigenvectors = torch.linalg.eigh(covariance)
            normals.append(eigenvectors[..., 0])
    return torch.cat(normals, dim=1).detach()


def _point_to_plane_per_cloud(
    reconstruction: Tensor,
    source: Tensor,
    source_normals: Tensor,
    *,
    chunk_size: int,
) -> Tensor:
    """Symmetric source-normal point-to-plane loss, one value per cloud."""

    if source.shape != source_normals.shape:
        raise ValueError("source and source_normals must have matching shapes")

    forward_parts = []
    for start in range(0, reconstruction.shape[1], chunk_size):
        query = reconstruction[:, start : start + chunk_size]
        indices = pairwise_squared(query, source).argmin(dim=2)
        matched_points = source.gather(1, indices[:, :, None].expand(-1, -1, 3))
        matched_normals = source_normals.gather(
            1, indices[:, :, None].expand(-1, -1, 3)
        )
        forward_parts.append(
            ((query - matched_points) * matched_normals).sum(dim=2).square()
        )
    backward_parts = []
    for start in range(0, source.shape[1], chunk_size):
        query = source[:, start : start + chunk_size]
        query_normals = source_normals[:, start : start + chunk_size]
        indices = pairwise_squared(query, reconstruction).argmin(dim=2)
        matched = reconstruction.gather(1, indices[:, :, None].expand(-1, -1, 3))
        backward_parts.append(((query - matched) * query_normals).sum(dim=2).square())
    forward = torch.cat(forward_parts, dim=1).mean(dim=1)
    backward = torch.cat(backward_parts, dim=1).mean(dim=1)
    return 0.5 * (forward + backward)


def _validate_context(context: ObjectiveContext, *, require_features: bool) -> None:
    if context.source_points.ndim != 3 or context.source_points.shape[-1] != 3:
        raise ValueError("source_points must have shape (batch, N, 3)")
    if context.num_output_points < 1 or context.distance_chunk_size < 1:
        raise ValueError("output-point and chunk sizes must be positive")
    if any(parameter.requires_grad for parameter in context.decoder.parameters()):
        raise ValueError("objective search requires a frozen decoder")
    if require_features:
        if context.feature_extractor is None:
            raise ValueError("feature objective requires a frozen feature extractor")
        if any(
            parameter.requires_grad
            for parameter in context.feature_extractor.parameters()
        ):
            raise ValueError("feature objective requires a frozen feature extractor")


def _decode(context: ObjectiveContext, coordinates: Tensor) -> Tensor:
    return context.decoder(
        coordinates,
        num_output_points=context.num_output_points,
    )


def _chamfer_builder(context: ObjectiveContext) -> ObjectiveScorer:
    _validate_context(context, require_features=False)

    def score(coordinates: Tensor) -> Tensor:
        return _per_cloud_chamfer(
            _decode(context, coordinates),
            context.source_points,
            chunk_size=context.distance_chunk_size,
        )

    return score


def _point_to_plane_builder(context: ObjectiveContext) -> ObjectiveScorer:
    _validate_context(context, require_features=False)
    normals = estimate_source_normals_pca(
        context.source_points,
        neighbors=context.normal_neighbors,
        chunk_size=context.normal_chunk_size,
    )

    def score(coordinates: Tensor) -> Tensor:
        return _point_to_plane_per_cloud(
            _decode(context, coordinates),
            context.source_points,
            normals,
            chunk_size=context.distance_chunk_size,
        )

    return score


def _feature_builder(context: ObjectiveContext) -> ObjectiveScorer:
    _validate_context(context, require_features=True)
    assert context.feature_extractor is not None
    with torch.no_grad():
        source_features = context.feature_extractor.extract_features(
            context.source_points
        ).detach()

    def score(coordinates: Tensor) -> Tensor:
        reconstruction_features = context.feature_extractor.extract_features(
            _decode(context, coordinates)
        )
        return (reconstruction_features - source_features).square().mean(dim=1)

    return score


def _mixed_builder(context: ObjectiveContext) -> ObjectiveScorer:
    _validate_context(context, require_features=True)
    if not 0.0 <= context.mixed_chamfer_weight <= 1.0:
        raise ValueError("mixed_chamfer_weight must be in [0, 1]")
    assert context.feature_extractor is not None
    with torch.no_grad():
        source_features = context.feature_extractor.extract_features(
            context.source_points
        ).detach()

    def score(coordinates: Tensor) -> Tensor:
        reconstruction = _decode(context, coordinates)
        chamfer = _per_cloud_chamfer(
            reconstruction,
            context.source_points,
            chunk_size=context.distance_chunk_size,
        )
        reconstruction_features = context.feature_extractor.extract_features(
            reconstruction
        )
        feature = (reconstruction_features - source_features).square().mean(dim=1)
        weight = context.mixed_chamfer_weight
        return weight * chamfer + (1.0 - weight) * feature

    return score


OBJECTIVE_REGISTRY: Mapping[str, ObjectiveBuilder] = {
    "chamfer": _chamfer_builder,
    "point_to_plane": _point_to_plane_builder,
    "feature_matching": _feature_builder,
    "mixed": _mixed_builder,
}


def build_objective_scorer(name: str, context: ObjectiveContext) -> ObjectiveScorer:
    """Build one registered scorer from source-only encode-time inputs."""

    try:
        builder = OBJECTIVE_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown objective: {name}") from error
    return builder(context)


@dataclass(frozen=True)
class ObjectiveSweepRegime:
    """One N/K stabilized-decoder dependency."""

    name: str
    num_points: int
    constellation_size: int
    stability_config: str
    stability_artifact_dir: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("regime name cannot be empty")
        if self.num_points < 8:
            raise ValueError("regime num_points must be at least eight")
        if not 2 <= self.constellation_size <= self.num_points:
            raise ValueError("regime constellation_size must be between 2 and N")


@dataclass(frozen=True)
class ObjectiveSweepExperimentConfig:
    """Validated multi-regime source-objective comparison configuration."""

    regimes: tuple[ObjectiveSweepRegime, ...]
    pc_error_executable: str
    objectives: tuple[str, ...] = OBJECTIVE_NAMES
    required_coordinate_bits: int = 12
    position_bits: int = 12
    timeout_seconds: float = 120.0
    decoder_seeds: tuple[int, ...] = (7, 17, 29, 41, 53, 67)
    start_methods: tuple[str, ...] = ("fps", "kmeans")
    decoder_evaluation_budget: int = 64
    adam_learning_rate: float = 0.03
    mixed_chamfer_weight: float = 0.5
    normal_neighbors: int = 16
    normal_chunk_size: int = 256
    selection_seed: int = 20_260_833
    splits: tuple[str, ...] = SUPPORTED_SPLITS
    max_clouds_per_split: int | None = None
    batch_size: int = 4
    classifier_regime: str = "k8_n2048"
    classifier_checkpoint: str | None = None
    expected_classifier_sha256: str | None = None
    classifier_seed: int = 20_260_846
    classifier_epochs: int = 20
    classifier_learning_rate: float = 1e-3
    classifier_feature_width: int = 64
    classifier_feature_dim: int = 128
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_833
    confidence_level: float = 0.95
    gate_accuracy_delta: float = 0.0
    gate_max_d1_d2_regression_percent: float = 2.0
    gate_min_source_accuracy: float = 0.7
    gate_min_regimes: int = 2
    output_dir: str = "artifacts/local/experiment_033_objective_sweep"

    def __post_init__(self) -> None:
        if not self.regimes or len({regime.name for regime in self.regimes}) != len(
            self.regimes
        ):
            raise ValueError("regimes must be nonempty and uniquely named")
        if self.classifier_regime not in {regime.name for regime in self.regimes}:
            raise ValueError("classifier_regime must name a configured regime")
        if self.normal_neighbors >= min(regime.num_points for regime in self.regimes):
            raise ValueError("normal_neighbors must be smaller than every regime N")
        if (
            not self.objectives
            or len(set(self.objectives)) != len(self.objectives)
            or set(self.objectives) - set(OBJECTIVE_REGISTRY)
        ):
            raise ValueError("objectives must be unique registered objective names")
        if "chamfer" not in self.objectives:
            raise ValueError("objectives must include the Chamfer control")
        if not 2 <= self.required_coordinate_bits <= 24:
            raise ValueError("required_coordinate_bits must be between 2 and 24")
        if not 2 <= self.position_bits <= 24:
            raise ValueError("position_bits must be between 2 and 24")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if len(self.decoder_seeds) < 2 or len(set(self.decoder_seeds)) != len(
            self.decoder_seeds
        ):
            raise ValueError("decoder_seeds must contain at least two unique seeds")
        if (
            not self.start_methods
            or len(set(self.start_methods)) != len(self.start_methods)
            or set(self.start_methods) - set(SELECTION_METHODS)
        ):
            raise ValueError("start_methods must be unique registered selectors")
        if self.decoder_evaluation_budget < 1 or self.adam_learning_rate <= 0:
            raise ValueError("Adam budget and learning rate must be positive")
        if not 0.0 <= self.mixed_chamfer_weight <= 1.0:
            raise ValueError("mixed_chamfer_weight must be in [0, 1]")
        if self.normal_neighbors < 3 or self.normal_chunk_size < 1:
            raise ValueError("PCA normal parameters are invalid")
        if not 0 <= self.selection_seed < 2**63:
            raise ValueError("selection_seed must be a nonnegative 63-bit integer")
        if (
            not self.splits
            or len(set(self.splits)) != len(self.splits)
            or set(self.splits) - set(SUPPORTED_SPLITS)
        ):
            raise ValueError("splits must be unique validation and/or ood entries")
        if self.max_clouds_per_split is not None and self.max_clouds_per_split < 1:
            raise ValueError("max_clouds_per_split must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.classifier_epochs < 1 or self.classifier_learning_rate <= 0:
            raise ValueError("classifier training parameters must be positive")
        if min(self.classifier_feature_width, self.classifier_feature_dim) < 4:
            raise ValueError("classifier feature dimensions must be at least four")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")
        if self.gate_accuracy_delta < 0:
            raise ValueError("gate_accuracy_delta cannot be negative")
        if self.gate_max_d1_d2_regression_percent < 0:
            raise ValueError("gate distortion tolerance cannot be negative")
        if not 0.0 <= self.gate_min_source_accuracy <= 1.0:
            raise ValueError("gate_min_source_accuracy must be in [0, 1]")
        if self.gate_min_regimes < 1:
            raise ValueError("gate_min_regimes must be positive")

    @classmethod
    def from_json(cls, path: Path) -> ObjectiveSweepExperimentConfig:
        values = json.loads(path.read_text())
        values["regimes"] = tuple(
            ObjectiveSweepRegime(**regime) for regime in values["regimes"]
        )
        for key in ("objectives", "decoder_seeds", "start_methods", "splits"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class _LoadedRegime:
    spec: ObjectiveSweepRegime
    stability: StabilityExperimentConfig
    datasets: Mapping[str, Any]
    data_protocol: Mapping[str, Any]
    stability_config_sha256: str
    stability_metrics_sha256: str


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _sample_tensor(sample: Mapping[str, Any], *names: str) -> Tensor | None:
    for name in names:
        value = sample.get(name)
        if isinstance(value, Tensor):
            return value
    return None


def _source_tensor(sample: Mapping[str, Any]) -> Tensor:
    source = _sample_tensor(sample, "source_points", "points")
    if source is None:
        raise ValueError("sample is missing source points")
    return source


def _sample_label(sample: Mapping[str, Any]) -> str:
    for name in ("category", "family"):
        if name in sample:
            value = sample[name]
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    raise ValueError("sample label tensor must be scalar")
                value = value.item()
            return str(value)
    raise ValueError("classifier sample is missing category/family")


def _metadata(sample: Mapping[str, Any], fallback: int) -> dict[str, Any]:
    def value(name: str, default: Any) -> Any:
        item = sample.get(name, default)
        if isinstance(item, Tensor):
            if item.numel() != 1:
                raise ValueError(f"sample metadata {name} must be scalar")
            return item.item()
        return item

    sample_id = int(value("sample_id", fallback))
    return {
        "family": str(value("family", value("category", "unknown"))),
        "model_id": str(value("model_id", sample_id)),
        "sample_id": sample_id,
    }


def _encoding_identity(sample: Mapping[str, Any], fallback: int) -> dict[str, Any]:
    """Return non-semantic identity fields permitted before objective search."""

    def value(name: str, default: Any) -> Any:
        item = sample.get(name, default)
        if isinstance(item, Tensor):
            if item.numel() != 1:
                raise ValueError(f"sample identity {name} must be scalar")
            return item.item()
        return item

    sample_id = int(value("sample_id", fallback))
    return {
        "model_id": str(value("model_id", sample_id)),
        "sample_id": sample_id,
    }


def _load_regime(spec: ObjectiveSweepRegime, required_bits: int) -> _LoadedRegime:
    stability_path = Path(spec.stability_config)
    artifact_dir = Path(spec.stability_artifact_dir)
    if not stability_path.is_file():
        raise FileNotFoundError(
            f"missing stabilized-decoder config for regime {spec.name}: "
            f"{stability_path}"
        )
    metrics_path = artifact_dir / "stability_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"missing stabilized-decoder artifact for regime {spec.name}: "
            f"{metrics_path}"
        )
    stability = StabilityExperimentConfig.from_json(stability_path)
    if (
        stability.num_points != spec.num_points
        or stability.constellation_size != spec.constellation_size
    ):
        raise RuntimeError(f"regime {spec.name} N/K differs from its stability config")
    if stability.coordinate_bits != required_bits:
        raise RuntimeError(
            f"regime {spec.name} uses q={stability.coordinate_bits}, expected "
            f"q={required_bits}"
        )
    if spec.constellation_size not in stability.training_constellation_sizes:
        raise RuntimeError(f"regime {spec.name} K was not seen in decoder training")
    metrics = json.loads(metrics_path.read_text())
    contract_checks = metrics.get("contract_checks")
    if (
        not isinstance(contract_checks, dict)
        or not contract_checks
        or not all(contract_checks.values())
    ):
        raise RuntimeError(f"regime {spec.name} stability contract failed")
    datasets = _datasets(stability)
    protocol = _data_protocol(stability, datasets)
    if protocol != metrics.get("data_protocol"):
        raise RuntimeError(f"regime {spec.name} data identity changed")
    return _LoadedRegime(
        spec=spec,
        stability=stability,
        datasets=datasets,
        data_protocol=protocol,
        stability_config_sha256=file_sha256(stability_path),
        stability_metrics_sha256=file_sha256(metrics_path),
    )


def _classifier_checkpoint_path(
    config: ObjectiveSweepExperimentConfig, output_dir: Path
) -> Path:
    if config.classifier_checkpoint is not None:
        return Path(config.classifier_checkpoint)
    return output_dir / "pointnet_classifier.pt"


def _train_classifier(
    dataset: Any,
    *,
    config: ObjectiveSweepExperimentConfig,
    training_membership_sha256: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PointNetClassifier, dict[str, int], dict[str, Any]]:
    labels = sorted({_sample_label(dataset[index]) for index in range(len(dataset))})
    class_to_index = {label: index for index, label in enumerate(labels)}
    set_seed(config.classifier_seed)
    classifier = PointNetClassifier(
        len(labels),
        feature_width=config.classifier_feature_width,
        feature_dim=config.classifier_feature_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=config.classifier_learning_rate
    )
    history = []
    for epoch in range(config.classifier_epochs):
        generator = torch.Generator().manual_seed(config.classifier_seed + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        total_loss = 0.0
        correct = 0
        count = 0
        classifier.train()
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            samples = [dataset[index] for index in batch_indices]
            points = torch.stack([_source_tensor(sample) for sample in samples]).to(
                device
            )
            targets = torch.tensor(
                [class_to_index[_sample_label(sample)] for sample in samples],
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(points)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(samples)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            count += len(samples)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": total_loss / count,
                "training_accuracy": correct / count,
            }
        )
    classifier.eval().requires_grad_(False)
    state_hash = _state_hash(classifier)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": classifier.state_dict(),
            "model_state_hash": state_hash,
            "class_to_index": class_to_index,
            "feature_width": config.classifier_feature_width,
            "feature_dim": config.classifier_feature_dim,
            "classifier_seed": config.classifier_seed,
            "classifier_epochs": config.classifier_epochs,
            "classifier_learning_rate": config.classifier_learning_rate,
            "training_membership_sha256": training_membership_sha256,
            "training_labels_only": True,
            "history": history,
        },
        checkpoint_path,
    )
    return (
        classifier,
        class_to_index,
        {
            "trained_now": True,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "model_state_hash": state_hash,
            "class_to_index": class_to_index,
            "training_membership_sha256": training_membership_sha256,
            "history": history,
        },
    )


def _load_classifier(
    checkpoint_path: Path,
    *,
    config: ObjectiveSweepExperimentConfig,
    training_membership_sha256: str,
    device: torch.device,
) -> tuple[PointNetClassifier, dict[str, int], dict[str, Any]]:
    actual_sha256 = file_sha256(checkpoint_path)
    if (
        config.expected_classifier_sha256 is not None
        and actual_sha256 != config.expected_classifier_sha256
    ):
        raise RuntimeError("classifier checkpoint hash differs from configuration")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("training_membership_sha256") != training_membership_sha256:
        raise RuntimeError("classifier was trained on a different training partition")
    if checkpoint.get("training_labels_only") is not True:
        raise RuntimeError("classifier checkpoint lacks its training-only contract")
    class_to_index = {
        str(label): int(index) for label, index in checkpoint["class_to_index"].items()
    }
    if set(class_to_index.values()) != set(range(len(class_to_index))):
        raise RuntimeError("classifier class indices must be contiguous")
    if (
        int(checkpoint["feature_width"]) != config.classifier_feature_width
        or int(checkpoint["feature_dim"]) != config.classifier_feature_dim
    ):
        raise RuntimeError("classifier architecture differs from configuration")
    classifier = PointNetClassifier(
        len(class_to_index),
        feature_width=int(checkpoint["feature_width"]),
        feature_dim=int(checkpoint["feature_dim"]),
    ).to(device)
    classifier.load_state_dict(checkpoint["model"])
    classifier.eval().requires_grad_(False)
    state_hash = _state_hash(classifier)
    if state_hash != checkpoint["model_state_hash"]:
        raise RuntimeError("classifier state differs from its checkpoint hash")
    return (
        classifier,
        class_to_index,
        {
            "trained_now": False,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": actual_sha256,
            "model_state_hash": state_hash,
            "class_to_index": class_to_index,
            "training_membership_sha256": training_membership_sha256,
            "history": checkpoint.get("history", []),
        },
    )


def _prepare_classifier(
    regime: _LoadedRegime,
    config: ObjectiveSweepExperimentConfig,
    *,
    output_dir: Path,
    device: torch.device,
) -> tuple[PointNetClassifier, dict[str, int], dict[str, Any]]:
    membership = str(regime.data_protocol["partitions"]["train"]["sha256"])
    checkpoint_path = _classifier_checkpoint_path(config, output_dir)
    if checkpoint_path.is_file():
        return _load_classifier(
            checkpoint_path,
            config=config,
            training_membership_sha256=membership,
            device=device,
        )
    if config.classifier_checkpoint is not None:
        raise FileNotFoundError(
            f"classifier checkpoint does not exist: {checkpoint_path}"
        )
    classifier, class_to_index, record = _train_classifier(
        regime.datasets["train"],
        config=config,
        training_membership_sha256=membership,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    if (
        config.expected_classifier_sha256 is not None
        and record["checkpoint_sha256"] != config.expected_classifier_sha256
    ):
        raise RuntimeError("trained classifier hash differs from configuration")
    return classifier, class_to_index, record


def _official_config(
    regime: _LoadedRegime, config: ObjectiveSweepExperimentConfig
) -> OfficialStabilityConfig:
    return OfficialStabilityConfig(
        stability_config=regime.spec.stability_config,
        stability_artifact_dir=regime.spec.stability_artifact_dir,
        pc_error_executable=config.pc_error_executable,
        position_bits=config.position_bits,
        timeout_seconds=config.timeout_seconds,
        decoder_seeds=config.decoder_seeds,
        refiner_seeds=regime.stability.refiner_seeds,
        splits=config.splits,
        max_clouds_per_split=None,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
        output_dir=config.output_dir,
    )


def _selection_seed(
    config: ObjectiveSweepExperimentConfig,
    *,
    regime: str,
    method: str,
    source_points: Tensor,
) -> int:
    digest = hashlib.sha256(
        json.dumps(
            [config.selection_seed, regime, method], separators=(",", ":")
        ).encode()
    )
    canonical = _canonical_source_points(source_points)
    digest.update(canonical.detach().cpu().contiguous().numpy().tobytes())
    return int.from_bytes(digest.digest()[:8], "big") % (2**63)


def _canonical_source_points(source_points: Tensor) -> Tensor:
    """Return a lexicographic source ordering for invariant initialization."""

    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("source_points must have shape (N, 3)")
    values = source_points.detach().cpu().numpy()
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0])).copy()
    indices = torch.from_numpy(order).to(source_points.device)
    return source_points[indices]


def _initial_starts(
    source: Tensor,
    regime: _LoadedRegime,
    config: ObjectiveSweepExperimentConfig,
) -> tuple[dict[str, Tensor], float]:
    started = time.perf_counter()
    starts = {}
    for method in config.start_methods:
        candidates = []
        for cloud in source:
            canonical = _canonical_source_points(cloud)
            candidates.append(
                SELECTION_METHODS[method](
                    canonical,
                    regime.spec.constellation_size,
                    regime.stability.coordinate_bits,
                    _selection_seed(
                        config,
                        regime=regime.spec.name,
                        method=method,
                        source_points=canonical,
                    ),
                    None,
                )
            )
        starts[method] = torch.stack(candidates)
    return starts, time.perf_counter() - started


def _search_objective(
    objective: str,
    context: ObjectiveContext,
    starts: Mapping[str, Tensor],
    config: ObjectiveSweepExperimentConfig,
) -> tuple[Tensor, Tensor, list[str], int, float]:
    scorer = build_objective_scorer(objective, context)
    results: dict[str, SearchResult] = {}
    started = time.perf_counter()
    for name, initial in starts.items():
        results[name] = adam_ste_search(
            scorer,
            initial,
            bits=config.required_coordinate_bits,
            decoder_evaluation_budget=config.decoder_evaluation_budget,
            learning_rate=config.adam_learning_rate,
        )
    elapsed = time.perf_counter() - started
    losses = torch.stack([results[name].losses for name in starts])
    best_losses, best_indices = losses.min(dim=0)
    names = tuple(starts)
    selected_names = [names[int(index)] for index in best_indices]
    coordinates = torch.stack(
        [
            results[selected_name].coordinates[cloud_index]
            for cloud_index, selected_name in enumerate(selected_names)
        ]
    )
    evaluations = config.decoder_evaluation_budget * len(starts)
    return coordinates, best_losses, selected_names, evaluations, elapsed


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["regime"],
        row["split"],
        row["objective"],
        row["decoder_seed"],
        row["model_id"],
        row["sample_id"],
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("objective-sweep JSONL contains duplicate rows")
    return rows


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _metric_cache_key(
    row: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        row["regime"],
        row["split"],
        row["decoder_seed"],
        row["model_id"],
        row["sample_id"],
        row["stream_sha256"],
    )


def _metric_cache(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    cache = {}
    for row in rows:
        cache[_metric_cache_key(row)] = {
            key: value
            for key, value in row.items()
            if key.startswith("d1_")
            or key.startswith("d2_")
            or key == "official_metric_seconds"
        }
    return cache


def _evaluation_tensors(
    samples: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[str]]:
    source = []
    fresh = []
    source_normals_rows = []
    fresh_normals_rows = []
    labels = []
    for sample in samples:
        source_points = _source_tensor(sample)
        fresh_points = _sample_tensor(sample, "fresh_points")
        source_normals = _sample_tensor(sample, "source_normals", "normals")
        fresh_normals = _sample_tensor(sample, "fresh_normals", "target_normals")
        if source_normals is None:
            raise ValueError("evaluation sample is missing source normals")
        source.append(source_points)
        fresh.append(source_points if fresh_points is None else fresh_points)
        source_normals_rows.append(source_normals)
        fresh_normals_rows.append(
            source_normals if fresh_normals is None else fresh_normals
        )
        labels.append(_sample_label(sample))
    return (
        torch.stack(source).to(device),
        torch.stack(fresh).to(device),
        torch.stack(source_normals_rows).to(device),
        torch.stack(fresh_normals_rows).to(device),
        labels,
    )


def _evaluate_coordinates(
    coordinates: Tensor,
    objective_losses: Tensor,
    selected_starts: Sequence[str],
    samples: Sequence[Mapping[str, Any]],
    *,
    objective: str,
    decoder: nn.Module,
    decoder_seed: int,
    regime: _LoadedRegime,
    split: str,
    encode_seconds_per_cloud: float,
    decoder_evaluations: int,
    classifier: PointNetClassifier,
    class_to_index: Mapping[str, int],
    classifier_state_hash: str,
    config: ObjectiveSweepExperimentConfig,
    scratch_root: Path,
    metric_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata = [_metadata(sample, index) for index, sample in enumerate(samples)]
    source, fresh, source_normals, fresh_normals, labels = _evaluation_tensors(
        samples, coordinates.device
    )
    decoded, streams, round_trip, lattice_exact, serialization_seconds = (
        _serialize_batch(coordinates, stability=regime.stability, mode="free")
    )
    with torch.no_grad():
        reconstruction = decoder(decoded, num_output_points=regime.stability.num_points)
        source_chamfer = _per_cloud_chamfer(
            reconstruction,
            source,
            chunk_size=regime.stability.distance_chunk_size,
        )
        fresh_chamfer = _per_cloud_chamfer(
            reconstruction,
            fresh,
            chunk_size=regime.stability.distance_chunk_size,
        )
        source_predictions = classifier(source).argmax(dim=1)
        reconstruction_predictions = classifier(reconstruction).argmax(dim=1)
    rows = []
    serialization_per_cloud = serialization_seconds / len(samples)
    for index, (cloud_metadata, stream, label) in enumerate(
        zip(metadata, streams, labels, strict=True)
    ):
        stream_sha256 = hashlib.sha256(stream).hexdigest()
        cache_probe = {
            "regime": regime.spec.name,
            "split": split,
            "decoder_seed": decoder_seed,
            **cloud_metadata,
            "stream_sha256": stream_sha256,
        }
        cache_key = _metric_cache_key(cache_probe)
        if cache_key not in metric_cache:
            with tempfile.TemporaryDirectory(
                prefix=f"{regime.spec.name}-{split}-d{decoder_seed}-",
                dir=scratch_root,
            ) as temporary:
                official = run_pc_error(
                    Path(config.pc_error_executable),
                    source[index].detach().cpu().numpy(),
                    reconstruction[index].detach().cpu().numpy(),
                    source_normals[index].detach().cpu().numpy(),
                    work_dir=Path(temporary),
                    position_bits=config.position_bits,
                    timeout_seconds=config.timeout_seconds,
                )
            metric_cache[cache_key] = {
                **official.metrics,
                "official_metric_seconds": official.elapsed_seconds,
            }
        fresh_geometry = standardized_geometry_metrics(
            reconstruction[index],
            fresh[index],
            fresh_normals[index],
            chunk_size=regime.stability.distance_chunk_size,
        )
        known_label = label in class_to_index
        target_index = class_to_index.get(label)
        rows.append(
            {
                "experiment": "033_encoder_objective_sweep",
                "regime": regime.spec.name,
                "split": split,
                "objective": objective,
                "decoder_seed": decoder_seed,
                **cloud_metadata,
                "num_points": regime.spec.num_points,
                "constellation_size": regime.spec.constellation_size,
                "coordinate_bits": regime.stability.coordinate_bits,
                "representation_class": "free-coordinate",
                "bitstream_mode": "free",
                "stream_hex": stream.hex(),
                "stream_sha256": stream_sha256,
                "stream_bytes": len(stream),
                "actual_stream_bits": 8 * len(stream),
                "actual_stream_bpp": 8.0 * len(stream) / regime.spec.num_points,
                "selected_start": selected_starts[index],
                "search_objective_value": float(objective_losses[index].item()),
                "decoder_evaluations": decoder_evaluations,
                "encode_seconds": encode_seconds_per_cloud + serialization_per_cloud,
                "source_chamfer_mse": float(source_chamfer[index].item()),
                "fresh_chamfer_mse": float(fresh_chamfer[index].item()),
                "fresh_d1_mse_proxy": fresh_geometry["d1_mse_proxy"],
                "fresh_d2_mse_proxy": fresh_geometry["d2_mse_proxy"],
                "fresh_p95_euclidean": fresh_geometry["p95_euclidean"],
                "fresh_p99_euclidean": fresh_geometry["p99_euclidean"],
                "fresh_hausdorff": fresh_geometry["hausdorff"],
                "fresh_sliced_wasserstein_rms_proxy": fresh_geometry[
                    "sliced_wasserstein_rms_proxy"
                ],
                "surface_measurement": (
                    "independent_finite_mesh_resample_proxy"
                    if _sample_tensor(samples[index], "fresh_points") is not None
                    else "finite_source_sample_proxy"
                ),
                "classifier_label": label,
                "classifier_label_known": known_label,
                "source_downstream_correct": (
                    bool(source_predictions[index].item() == target_index)
                    if known_label
                    else None
                ),
                "reconstruction_downstream_correct": (
                    bool(reconstruction_predictions[index].item() == target_index)
                    if known_label
                    else None
                ),
                "classifier_state_hash": classifier_state_hash,
                "source_only_optimization": True,
                "objective_inputs": [
                    "source_points",
                    "frozen_decoder",
                    *(
                        ["frozen_feature_extractor"]
                        if objective in {"feature_matching", "mixed"}
                        else []
                    ),
                    *(
                        ["source_pca_knn_normals"]
                        if objective == "point_to_plane"
                        else []
                    ),
                ],
                "labels_used_after_encoding_only": True,
                "provided_normals_used_after_encoding_only": True,
                "serialized_round_trip_exact": round_trip,
                "coordinates_on_exact_lattice": lattice_exact,
                **metric_cache[cache_key],
            }
        )
    return rows


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _paired_interval(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    config: ObjectiveSweepExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    def identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["decoder_seed"],
            row["family"],
            row["model_id"],
            row["sample_id"],
        )

    baseline_map = {identity(row): row for row in baseline}
    candidate_map = {identity(row): row for row in candidate}
    if baseline_map.keys() != candidate_map.keys() or not baseline_map:
        raise RuntimeError("objective rows do not form a complete paired comparison")
    pairs = [(baseline_map[key], candidate_map[key]) for key in sorted(baseline_map)]
    known = [
        index
        for index, (first, second) in enumerate(pairs)
        if first["classifier_label_known"] and second["classifier_label_known"]
    ]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pairs), size=(config.bootstrap_samples, len(pairs)))
    base_d1 = np.asarray([pair[0]["d1_mse"] for pair in pairs], dtype=np.float64)
    cand_d1 = np.asarray([pair[1]["d1_mse"] for pair in pairs], dtype=np.float64)
    base_d2 = np.asarray([pair[0]["d2_mse"] for pair in pairs], dtype=np.float64)
    cand_d2 = np.asarray([pair[1]["d2_mse"] for pair in pairs], dtype=np.float64)

    def regression(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_rmse = np.sqrt(first[draws].mean(axis=1))
        second_rmse = np.sqrt(second[draws].mean(axis=1))
        return 100.0 * (second_rmse - first_rmse) / np.maximum(first_rmse, 1e-12)

    d1 = regression(base_d1, cand_d1)
    d2 = regression(base_d2, cand_d2)
    alpha = 0.5 * (1.0 - config.confidence_level)

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1.0 - alpha)),
        ]

    accuracy_delta: float | None = None
    accuracy_interval: list[float] | None = None
    if known:
        base_accuracy = np.asarray(
            [pairs[index][0]["reconstruction_downstream_correct"] for index in known],
            dtype=np.float64,
        )
        candidate_accuracy = np.asarray(
            [pairs[index][1]["reconstruction_downstream_correct"] for index in known],
            dtype=np.float64,
        )
        accuracy_delta = float((candidate_accuracy - base_accuracy).mean())
        accuracy_draws = rng.integers(
            0, len(known), size=(config.bootstrap_samples, len(known))
        )
        accuracy_interval = interval(
            (candidate_accuracy - base_accuracy)[accuracy_draws].mean(axis=1)
        )
    return {
        "paired_rows": len(pairs),
        "known_label_rows": len(known),
        "bootstrap_unit": "decoder-cloud cell",
        "confidence_level": config.confidence_level,
        "accuracy_delta": accuracy_delta,
        "accuracy_delta_interval": accuracy_interval,
        "d1_rmse_regression_percent": float(
            100.0
            * (math.sqrt(float(cand_d1.mean())) - math.sqrt(float(base_d1.mean())))
            / max(math.sqrt(float(base_d1.mean())), 1e-12)
        ),
        "d1_rmse_regression_percent_interval": interval(d1),
        "d2_rmse_regression_percent": float(
            100.0
            * (math.sqrt(float(cand_d2.mean())) - math.sqrt(float(base_d2.mean())))
            / max(math.sqrt(float(base_d2.mean())), 1e-12)
        ),
        "d2_rmse_regression_percent_interval": interval(d2),
    }


def summarize_objective_sweep_rows(
    rows: Sequence[Mapping[str, Any]], config: ObjectiveSweepExperimentConfig
) -> dict[str, Any]:
    """Build distortion/accuracy Pareto tables and the predeclared G-B2 gate."""

    points = []
    for regime in config.regimes:
        for split in config.splits:
            regime_rows = [
                row
                for row in rows
                if row["regime"] == regime.name and row["split"] == split
            ]
            for objective in config.objectives:
                selected = [row for row in regime_rows if row["objective"] == objective]
                if not selected:
                    continue
                accuracy = _mean_or_none(
                    [row["reconstruction_downstream_correct"] for row in selected]
                )
                source_accuracy = _mean_or_none(
                    [row["source_downstream_correct"] for row in selected]
                )
                points.append(
                    {
                        "regime": regime.name,
                        "num_points": regime.num_points,
                        "constellation_size": regime.constellation_size,
                        "split": split,
                        "objective": objective,
                        "rows": len(selected),
                        "actual_stream_bits": int(selected[0]["actual_stream_bits"]),
                        "mean_encode_seconds": float(
                            np.mean([row["encode_seconds"] for row in selected])
                        ),
                        "source_chamfer_rmse": math.sqrt(
                            float(
                                np.mean([row["source_chamfer_mse"] for row in selected])
                            )
                        ),
                        "fresh_chamfer_rmse": math.sqrt(
                            float(
                                np.mean([row["fresh_chamfer_mse"] for row in selected])
                            )
                        ),
                        "official_d1_rmse_grid_units": math.sqrt(
                            float(np.mean([row["d1_mse"] for row in selected]))
                        ),
                        "official_d2_rmse_grid_units": math.sqrt(
                            float(np.mean([row["d2_mse"] for row in selected]))
                        ),
                        "reconstruction_accuracy": accuracy,
                        "source_classifier_accuracy": source_accuracy,
                        "known_label_rows": sum(
                            bool(row["classifier_label_known"]) for row in selected
                        ),
                    }
                )
    for point in points:
        peers = [
            other
            for other in points
            if other["regime"] == point["regime"]
            and other["split"] == point["split"]
            and other["reconstruction_accuracy"] is not None
        ]
        if point["reconstruction_accuracy"] is None:
            point["on_distortion_accuracy_pareto"] = None
        else:
            point["on_distortion_accuracy_pareto"] = not any(
                other["official_d1_rmse_grid_units"]
                <= point["official_d1_rmse_grid_units"]
                and other["reconstruction_accuracy"] >= point["reconstruction_accuracy"]
                and (
                    other["official_d1_rmse_grid_units"]
                    < point["official_d1_rmse_grid_units"]
                    or other["reconstruction_accuracy"]
                    > point["reconstruction_accuracy"]
                )
                for other in peers
                if other is not point
            )

    comparisons = []
    for regime_index, regime in enumerate(config.regimes):
        baseline = [
            row
            for row in rows
            if row["regime"] == regime.name
            and row["split"] == "validation"
            and row["objective"] == "chamfer"
        ]
        if not baseline:
            continue
        source_accuracy = _mean_or_none(
            [row["source_downstream_correct"] for row in baseline]
        )
        for objective_index, objective in enumerate(config.objectives):
            if objective == "chamfer":
                continue
            candidate = [
                row
                for row in rows
                if row["regime"] == regime.name
                and row["split"] == "validation"
                and row["objective"] == objective
            ]
            comparison = _paired_interval(
                baseline,
                candidate,
                config=config,
                seed=config.bootstrap_seed + 1009 * regime_index + objective_index,
            )
            accuracy_interval = comparison["accuracy_delta_interval"]
            comparison.update(
                {
                    "regime": regime.name,
                    "num_points": regime.num_points,
                    "constellation_size": regime.constellation_size,
                    "baseline_objective": "chamfer",
                    "candidate_objective": objective,
                    "source_classifier_accuracy": source_accuracy,
                    "passes": bool(
                        source_accuracy is not None
                        and source_accuracy >= config.gate_min_source_accuracy
                        and accuracy_interval is not None
                        and accuracy_interval[0] > config.gate_accuracy_delta
                        and comparison["d1_rmse_regression_percent_interval"][1]
                        <= config.gate_max_d1_d2_regression_percent
                        and comparison["d2_rmse_regression_percent_interval"][1]
                        <= config.gate_max_d1_d2_regression_percent
                    ),
                }
            )
            comparisons.append(comparison)
    qualifying = {}
    for objective in config.objectives:
        accepted = [
            comparison
            for comparison in comparisons
            if comparison["candidate_objective"] == objective and comparison["passes"]
        ]
        if accepted:
            qualifying[objective] = accepted
    passes = any(
        len(accepted) >= config.gate_min_regimes
        and len({comparison["constellation_size"] for comparison in accepted}) >= 2
        and len({comparison["num_points"] for comparison in accepted}) >= 2
        for accepted in qualifying.values()
    )
    return {
        "pareto_tables": points,
        "g_b2": {
            "hypothesis": "Chamfer may be the wrong per-cloud encoder objective",
            "comparison": "non-Chamfer objective versus matched Chamfer control",
            "accuracy_delta_lower_bound_must_exceed": config.gate_accuracy_delta,
            "d1_d2_rmse_regression_upper_bound_percent": (
                config.gate_max_d1_d2_regression_percent
            ),
            "minimum_source_classifier_accuracy": config.gate_min_source_accuracy,
            "minimum_regimes": config.gate_min_regimes,
            "requires_two_k_and_two_n": True,
            "comparisons": comparisons,
            "passes": passes,
        },
    }


def run_objective_sweep_experiment(
    config: ObjectiveSweepExperimentConfig,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run all source-only objectives against matched stabilized decoders."""

    device = select_device(device_name)
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("objective sweep requires a torch CPU, MPS, or CUDA device")
    executable = Path(config.pc_error_executable)
    if not executable.is_file():
        raise FileNotFoundError(f"pc_error executable does not exist: {executable}")
    regimes = [
        _load_regime(spec, config.required_coordinate_bits) for spec in config.regimes
    ]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = output_dir / "metric_scratch"
    scratch_root.mkdir(exist_ok=True)
    classifier_regime = next(
        regime for regime in regimes if regime.spec.name == config.classifier_regime
    )
    classifier, class_to_index, classifier_record = _prepare_classifier(
        classifier_regime,
        config,
        output_dir=output_dir,
        device=device,
    )
    classifier_hash = _state_hash(classifier)

    dependencies = {
        regime.spec.name: {
            "stability_config_sha256": regime.stability_config_sha256,
            "stability_metrics_sha256": regime.stability_metrics_sha256,
            "data_protocol": regime.data_protocol,
        }
        for regime in regimes
    }
    manifest = {
        "experiment": "033_encoder_objective_sweep",
        "config": _json_ready(asdict(config)),
        "device": device.type,
        "pc_error_sha256": file_sha256(executable),
        "classifier_checkpoint_sha256": classifier_record["checkpoint_sha256"],
        "classifier_state_hash": classifier_hash,
        "dependencies": dependencies,
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("existing Experiment 033 run manifest differs")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rows_path = output_dir / "objective_sweep_per_cloud.jsonl"
    rows = _load_rows(rows_path)
    resumed_rows = len(rows)
    completed = {_row_key(row) for row in rows}
    metric_cache = _metric_cache(rows)
    model_records = []
    decoder_hashes_unchanged = True
    started = time.perf_counter()
    for regime in regimes:
        official = _official_config(regime, config)
        for decoder_seed in config.decoder_seeds:
            decoder, _, model_record = _load_models(
                regime.stability,
                official,
                decoder_seed=decoder_seed,
                refiner_seed=None,
                device=device,
            )
            decoder_hash = _state_hash(decoder)
            model_records.append(
                {
                    "regime": regime.spec.name,
                    "decoder_seed": decoder_seed,
                    **model_record,
                }
            )
            for split in config.splits:
                dataset = regime.datasets[split]
                count = len(dataset)
                if config.max_clouds_per_split is not None:
                    count = min(count, config.max_clouds_per_split)
                for batch_start in range(0, count, config.batch_size):
                    indices = range(
                        batch_start, min(batch_start + config.batch_size, count)
                    )
                    samples = [dataset[index] for index in indices]
                    encoding_metadata = [
                        _encoding_identity(sample, index)
                        for sample, index in zip(samples, indices, strict=True)
                    ]
                    expected_keys = {
                        (
                            regime.spec.name,
                            split,
                            objective,
                            decoder_seed,
                            cloud["model_id"],
                            cloud["sample_id"],
                        )
                        for objective in config.objectives
                        for cloud in encoding_metadata
                    }
                    if expected_keys <= completed:
                        continue
                    source = torch.stack(
                        [_source_tensor(sample) for sample in samples]
                    ).to(device)
                    starts, selection_seconds = _initial_starts(
                        source,
                        regime,
                        config,
                    )
                    context = ObjectiveContext(
                        decoder=decoder,
                        source_points=source,
                        num_output_points=regime.spec.num_points,
                        distance_chunk_size=regime.stability.distance_chunk_size,
                        normal_neighbors=config.normal_neighbors,
                        normal_chunk_size=config.normal_chunk_size,
                        mixed_chamfer_weight=config.mixed_chamfer_weight,
                        feature_extractor=classifier,
                    )
                    for objective in config.objectives:
                        coordinates, losses, selected_starts, evaluations, elapsed = (
                            _search_objective(objective, context, starts, config)
                        )
                        candidate_rows = _evaluate_coordinates(
                            coordinates,
                            losses,
                            selected_starts,
                            samples,
                            objective=objective,
                            decoder=decoder,
                            decoder_seed=decoder_seed,
                            regime=regime,
                            split=split,
                            encode_seconds_per_cloud=(selection_seconds + elapsed)
                            / len(samples),
                            decoder_evaluations=evaluations,
                            classifier=classifier,
                            class_to_index=class_to_index,
                            classifier_state_hash=classifier_hash,
                            config=config,
                            scratch_root=scratch_root,
                            metric_cache=metric_cache,
                        )
                        expected_bytes = expected_stream_bytes(
                            regime.spec.constellation_size,
                            regime.stability.coordinate_bits,
                        )
                        for row in candidate_rows:
                            key = _row_key(row)
                            if key in completed:
                                continue
                            if row["stream_bytes"] != expected_bytes:
                                raise RuntimeError(
                                    "Experiment 033 stream size is inconsistent"
                                )
                            _append_row(rows_path, row)
                            rows.append(row)
                            completed.add(key)
            if _state_hash(decoder) != decoder_hash:
                decoder_hashes_unchanged = False
                raise RuntimeError("frozen decoder changed during objective search")
    if _state_hash(classifier) != classifier_hash:
        raise RuntimeError("frozen feature extractor changed during objective search")

    expected_rows = 0
    for regime in regimes:
        for split in config.splits:
            count = len(regime.datasets[split])
            if config.max_clouds_per_split is not None:
                count = min(count, config.max_clouds_per_split)
            expected_rows += count * len(config.decoder_seeds) * len(config.objectives)
    summary = summarize_objective_sweep_rows(rows, config)
    contract_checks = {
        "complete_factorial": len(rows) == expected_rows,
        "decoder_hashes_unchanged": decoder_hashes_unchanged,
        "classifier_hash_unchanged": _state_hash(classifier) == classifier_hash,
        "source_only_optimization": bool(
            rows and all(row["source_only_optimization"] for row in rows)
        ),
        "labels_used_after_encoding_only": bool(
            rows and all(row["labels_used_after_encoding_only"] for row in rows)
        ),
        "provided_normals_used_after_encoding_only": bool(
            rows
            and all(row["provided_normals_used_after_encoding_only"] for row in rows)
        ),
        "exact_stream_round_trip": bool(
            rows and all(row["serialized_round_trip_exact"] for row in rows)
        ),
        "exact_coordinate_lattice": bool(
            rows and all(row["coordinates_on_exact_lattice"] for row in rows)
        ),
        "actual_streams_present": bool(
            rows and all(len(bytes.fromhex(row["stream_hex"])) > 0 for row in rows)
        ),
        "q_matches_configuration": bool(
            rows
            and all(
                row["coordinate_bits"] == config.required_coordinate_bits
                for row in rows
            )
        ),
    }
    result = {
        "experiment": "033_encoder_objective_sweep",
        "config": _json_ready(asdict(config)),
        "device": device.type,
        "resumed_rows": resumed_rows,
        "per_cloud_rows": len(rows),
        "expected_per_cloud_rows": expected_rows,
        "objective_registry": list(OBJECTIVE_REGISTRY),
        "objective_contract": (
            "each search scorer receives only source points, a frozen decoder, "
            "source-PCA normals when applicable, and a frozen training-split "
            "feature extractor; labels and evaluation targets are post-encode only"
        ),
        "contract_checks": contract_checks,
        "classifier": classifier_record,
        "model_records": model_records,
        "dependencies": dependencies,
        "statistics": summary,
        "elapsed_seconds": time.perf_counter() - started,
        "per_cloud_path": str(rows_path),
    }
    if not all(contract_checks.values()):
        raise RuntimeError("Experiment 033 scientific contract failed")
    (output_dir / "objective_sweep_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_033_objective_sweep_smoke.json"),
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--max-clouds-per-split", type=int)
    parser.add_argument("--pc-error", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = ObjectiveSweepExperimentConfig.from_json(args.config)
    if args.max_clouds_per_split is not None:
        config = replace(config, max_clouds_per_split=args.max_clouds_per_split)
    if args.pc_error is not None:
        config = replace(config, pc_error_executable=str(args.pc_error))
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    result = run_objective_sweep_experiment(config, device_name=args.device)
    print(
        json.dumps(
            {
                "rows": result["per_cloud_rows"],
                "gate_g_b2_passes": result["statistics"]["g_b2"]["passes"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(
                    Path(config.output_dir) / "objective_sweep_metrics.json"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
