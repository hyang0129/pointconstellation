"""Experiment 030: classification and retrieval from frozen representations.

The learned encoders in this module receive only the encoder-visible source
point cloud.  Categories are attached to cached representations only after the
encoder call returns, and are used solely by the downstream classifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset

from pointconstellation.bitstream import (
    decode_constellation,
    encode_constellation,
    expected_stream_bytes,
)
from pointconstellation.codecs import Tmc3RatePoint, run_tmc3
from pointconstellation.data import (
    MeshSurfaceDataset,
    ProceduralPointCloudDataset,
    file_sha256,
    load_mesh_manifest,
)
from pointconstellation.data.procedural import FAMILIES
from pointconstellation.feature_bitstream import (
    decode_features,
    encode_features,
    expected_feature_stream_bytes,
)
from pointconstellation.feature_codec_benchmark import FeatureCodecBenchmarkConfig
from pointconstellation.headroom_experiment import (
    _search_adam_start,
    _source_scorer,
)
from pointconstellation.models.feature_codec import VariableFeatureCodec
from pointconstellation.official_stability import (
    OfficialStabilityConfig,
    _load_models,
)
from pointconstellation.refiner_experiment import _fps, _state_hash
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
)
from pointconstellation.train import select_device, set_seed

SPLITS = ("train", "validation", "ood")
CACHE_VERSION = 1
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class DownstreamGpccConfig:
    """Nearest-actual-rate G-PCC control configuration."""

    executable: str
    rate_points: tuple[Tmc3RatePoint, ...]
    encoder_args: tuple[str, ...] = ()
    position_bits: int = 12
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not self.rate_points:
            raise ValueError("G-PCC requires at least one rate point")
        if len({point.name for point in self.rate_points}) != len(self.rate_points):
            raise ValueError("G-PCC rate-point names must be unique")
        if not 2 <= self.position_bits <= 24:
            raise ValueError("G-PCC position_bits must be between 2 and 24")
        if self.timeout_seconds <= 0:
            raise ValueError("G-PCC timeout_seconds must be positive")
        Tmc3RatePoint("common", self.encoder_args)
        common = {argument.split("=", 1)[0] for argument in self.encoder_args}
        for point in self.rate_points:
            options = {argument.split("=", 1)[0] for argument in point.encoder_args}
            overlap = common & options
            if overlap:
                raise ValueError(
                    f"G-PCC common/rate arguments overlap: {sorted(overlap)}"
                )


@dataclass(frozen=True)
class ClassifierConfig:
    """Common optimization budget for PointNet and feature-MLP classifiers."""

    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_width: int = 64
    embedding_dim: int = 64

    def __post_init__(self) -> None:
        if (
            min(
                self.epochs,
                self.batch_size,
                self.hidden_width,
                self.embedding_dim,
            )
            < 1
        ):
            raise ValueError("classifier counts and widths must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("classifier optimizer values are invalid")


@dataclass(frozen=True)
class DownstreamClassificationConfig:
    """Validated Experiment 030 extraction and evaluation protocol."""

    stability_config: str = "configs/experiment_019_stability_modelnet40.json"
    stability_artifact_dir: str = "artifacts/local/experiment_019_stability_modelnet40"
    official_stability_artifact_dir: str | None = (
        "artifacts/local/experiment_020_official_stability"
    )
    feature_config: str | None = "configs/experiment_018_feature_codec_multiseed.json"
    feature_artifact_dir: str | None = (
        "artifacts/local/experiment_018_feature_codec_multiseed"
    )
    dataset_root_override: str | None = None
    dataset_manifest_override: str | None = None
    feature_dataset_manifest_override: str | None = None
    constellation_sizes: tuple[int, ...] = (4, 8, 16, 32)
    feature_latent_dims: tuple[int, ...] = (20, 38, 74, 146)
    decoder_seed: int = 7
    refiner_seed: int = 101
    feature_model_seed: int = 7
    classifier_seeds: tuple[int, ...] = (3001, 3011, 3023)
    adam_budget: int = 64
    adam_learning_rate: float = 0.03
    primary_constellation_size: int = 8
    max_samples_per_category: int | None = None
    extraction_batch_size: int = 4
    classifier_epochs: int = 40
    classifier_batch_size: int = 32
    classifier_learning_rate: float = 1e-3
    classifier_weight_decay: float = 1e-4
    classifier_hidden_width: int = 64
    classifier_embedding_dim: int = 64
    retrieval_k: int = 10
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_830
    confidence_level: float = 0.95
    include_adam: bool = True
    include_refiner: bool = True
    include_fps: bool = True
    include_feature: bool = True
    include_source: bool = True
    gpcc: DownstreamGpccConfig | None = None
    output_dir: str = "artifacts/local/experiment_030_downstream"

    def __post_init__(self) -> None:
        if (
            not self.constellation_sizes
            or len(set(self.constellation_sizes)) != len(self.constellation_sizes)
            or tuple(sorted(self.constellation_sizes)) != self.constellation_sizes
            or min(self.constellation_sizes) < 2
        ):
            raise ValueError(
                "constellation_sizes must be unique, increasing, and at least two"
            )
        if self.primary_constellation_size not in self.constellation_sizes:
            raise ValueError("primary constellation size is absent from rate points")
        if self.include_feature and (
            self.feature_config is None or self.feature_artifact_dir is None
        ):
            raise ValueError("feature extraction requires config and artifact paths")
        if self.include_feature and (
            len(self.feature_latent_dims) != len(self.constellation_sizes)
            or len(set(self.feature_latent_dims)) != len(self.feature_latent_dims)
        ):
            raise ValueError("feature latent dimensions must align uniquely with K")
        if not self.classifier_seeds or len(set(self.classifier_seeds)) != len(
            self.classifier_seeds
        ):
            raise ValueError("classifier_seeds must be nonempty and unique")
        if min(self.decoder_seed, self.refiner_seed, self.feature_model_seed) < 0:
            raise ValueError("model seeds must be nonnegative")
        if self.adam_budget < 1 or self.adam_learning_rate <= 0:
            raise ValueError("Adam search budget and learning rate must be positive")
        if (
            self.max_samples_per_category is not None
            and self.max_samples_per_category < 1
        ):
            raise ValueError("max_samples_per_category must be positive")
        if (
            min(
                self.extraction_batch_size,
                self.classifier_epochs,
                self.classifier_batch_size,
                self.classifier_hidden_width,
                self.classifier_embedding_dim,
                self.retrieval_k,
            )
            < 1
        ):
            raise ValueError(
                "batch, epoch, retrieval, and width values must be positive"
            )
        if self.classifier_learning_rate <= 0 or self.classifier_weight_decay < 0:
            raise ValueError("classifier optimizer values are invalid")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

    @classmethod
    def from_json(cls, path: Path) -> DownstreamClassificationConfig:
        values = json.loads(path.read_text())
        for key in (
            "constellation_sizes",
            "feature_latent_dims",
            "classifier_seeds",
        ):
            if key in values:
                values[key] = tuple(values[key])
        gpcc_values = values.pop("gpcc", None)
        gpcc = None
        if gpcc_values is not None:
            if "encoder_args" in gpcc_values:
                gpcc_values["encoder_args"] = tuple(gpcc_values["encoder_args"])
            gpcc_values["rate_points"] = tuple(
                Tmc3RatePoint(
                    name=point["name"],
                    encoder_args=tuple(point["encoder_args"]),
                )
                for point in gpcc_values["rate_points"]
            )
            gpcc = DownstreamGpccConfig(**gpcc_values)
        return cls(gpcc=gpcc, **values)

    @property
    def classifier(self) -> ClassifierConfig:
        return ClassifierConfig(
            epochs=self.classifier_epochs,
            batch_size=self.classifier_batch_size,
            learning_rate=self.classifier_learning_rate,
            weight_decay=self.classifier_weight_decay,
            hidden_width=self.classifier_hidden_width,
            embedding_dim=self.classifier_embedding_dim,
        )


@dataclass(frozen=True)
class RepresentationSpec:
    """One downstream classifier input and its declared rate semantics."""

    key: str
    method: str
    view: str
    input_kind: str
    representation_class: str
    constellation_size: int | None
    target_stream_bytes: int | None
    rate_role: str = "coded"


@dataclass(frozen=True)
class ExtractedBatch:
    """Frozen, label-free representations produced for one source batch."""

    values: tuple[FloatArray, ...]
    stream_bytes: NDArray[np.float64]
    stream_sha256: tuple[str, ...]
    rate_points: tuple[str, ...]


@dataclass(frozen=True)
class CachedRepresentation:
    """A checked, aligned classifier input cache."""

    spec: RepresentationSpec
    values: FloatArray
    lengths: NDArray[np.int64]
    categories: NDArray[np.str_]
    model_ids: NDArray[np.str_]
    sample_ids: NDArray[np.int64]
    stream_bytes: NDArray[np.float64]
    stream_sha256: NDArray[np.str_]
    rate_points: NDArray[np.str_]

    def __post_init__(self) -> None:
        count = len(self.values)
        aligned = (
            self.lengths,
            self.categories,
            self.model_ids,
            self.sample_ids,
            self.stream_bytes,
            self.stream_sha256,
            self.rate_points,
        )
        if any(len(value) != count for value in aligned):
            raise ValueError("cached representation fields do not align")
        if self.spec.input_kind == "point_set":
            if self.values.ndim != 3 or self.values.shape[-1] != 3:
                raise ValueError("point-set cache must have shape (B, N, 3)")
            if np.any(self.lengths < 1) or np.any(self.lengths > self.values.shape[1]):
                raise ValueError("point-set cache contains invalid lengths")
        elif self.spec.input_kind == "feature":
            if self.values.ndim != 2:
                raise ValueError("feature cache must have shape (B, D)")
            if np.any(self.lengths != self.values.shape[1]):
                raise ValueError("feature cache lengths differ from feature width")
        else:
            raise ValueError(f"unknown classifier input kind: {self.spec.input_kind}")
        if not np.isfinite(self.values).all():
            raise ValueError("cached representations must be finite")


class SmallPointNet(nn.Module):
    """Small permutation-invariant classifier shared by all point-set views."""

    def __init__(
        self,
        num_classes: int,
        *,
        hidden_width: int = 64,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(hidden_width, embedding_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, points: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("PointNet inputs must have shape (B, N, 3)")
        embedded = self.point_mlp(points)
        positions = torch.arange(points.shape[1], device=points.device)[None]
        valid = positions < lengths[:, None]
        pooled = embedded.masked_fill(~valid[:, :, None], -torch.inf).amax(dim=1)
        features = self.embedding(pooled)
        return self.head(features), features


class FeatureMLP(nn.Module):
    """Matched-budget classifier for ordered feature-codec latents."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_width: int = 64,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, embedding_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        del lengths
        embedding = self.backbone(features)
        return self.head(embedding), embedding


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _coordinate_specs(
    config: DownstreamClassificationConfig,
    stability: StabilityExperimentConfig,
) -> list[RepresentationSpec]:
    specs = []
    for size in config.constellation_sizes:
        stream_bytes = expected_stream_bytes(size, stability.coordinate_bits)
        if config.include_fps:
            for view in ("raw", "decoded"):
                specs.append(
                    RepresentationSpec(
                        key=f"fps_{view}_k{size:04d}",
                        method="fps",
                        view=view,
                        input_kind="point_set",
                        representation_class="strict-subset",
                        constellation_size=size,
                        target_stream_bytes=stream_bytes,
                    )
                )
        if config.include_adam:
            for view in ("raw", "decoded"):
                specs.append(
                    RepresentationSpec(
                        key=f"adam{config.adam_budget}_{view}_k{size:04d}",
                        method="adam_ste",
                        view=view,
                        input_kind="point_set",
                        representation_class="free-coordinate",
                        constellation_size=size,
                        target_stream_bytes=stream_bytes,
                    )
                )
        if config.include_refiner and size == config.primary_constellation_size:
            for view in ("raw", "decoded"):
                specs.append(
                    RepresentationSpec(
                        key=f"refiner_{view}_k{size:04d}",
                        method="refiner",
                        view=view,
                        input_kind="point_set",
                        representation_class="free-coordinate",
                        constellation_size=size,
                        target_stream_bytes=stream_bytes,
                    )
                )
    return specs


def _representation_specs(
    config: DownstreamClassificationConfig,
    stability: StabilityExperimentConfig,
    feature: FeatureCodecBenchmarkConfig | None,
) -> tuple[RepresentationSpec, ...]:
    specs = _coordinate_specs(config, stability)
    if config.include_feature:
        assert feature is not None
        for size, latent_dim in zip(
            config.constellation_sizes, config.feature_latent_dims, strict=True
        ):
            coordinate_bytes = expected_stream_bytes(size, stability.coordinate_bits)
            feature_bytes = expected_feature_stream_bytes(
                latent_dim, feature.feature_bits
            )
            if feature_bytes != coordinate_bytes:
                raise ValueError(
                    f"feature latent D={latent_dim} is not byte-matched to K={size}"
                )
            specs.append(
                RepresentationSpec(
                    key=f"feature_latent_k{size:04d}",
                    method="feature_latent",
                    view="latent",
                    input_kind="feature",
                    representation_class="feature-ablation",
                    constellation_size=size,
                    target_stream_bytes=feature_bytes,
                )
            )
    if config.gpcc is not None:
        for size in config.constellation_sizes:
            specs.append(
                RepresentationSpec(
                    key=f"gpcc_decoded_k{size:04d}",
                    method="gpcc",
                    view="decoded",
                    input_kind="point_set",
                    representation_class="standard-codec",
                    constellation_size=size,
                    target_stream_bytes=expected_stream_bytes(
                        size, stability.coordinate_bits
                    ),
                )
            )
    if config.include_source:
        specs.append(
            RepresentationSpec(
                key="source_full",
                method="source",
                view="full",
                input_kind="point_set",
                representation_class="uncoded-upper-bound",
                constellation_size=None,
                target_stream_bytes=None,
                rate_role="upper_bound",
            )
        )
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("representation specifications contain duplicate keys")
    return tuple(specs)


def _serialize_coordinates(
    coordinates: Tensor,
    *,
    bits: int,
    mode: str,
    output_points: int,
) -> tuple[Tensor, NDArray[np.float64], tuple[str, ...]]:
    """Round-trip a batch through the canonical complete coordinate stream."""

    decoded = []
    stream_bytes = []
    hashes = []
    for row in coordinates.detach().cpu().numpy():
        stream = encode_constellation(
            row,
            bits=bits,
            mode=mode,
            output_points=output_points,
        )
        packet = decode_constellation(stream)
        repeated = encode_constellation(
            packet.coordinates,
            bits=packet.bits,
            mode=packet.mode,
            output_points=packet.output_points,
        )
        if repeated != stream:
            raise RuntimeError("coordinate representation failed exact round trip")
        levels = (1 << packet.bits) - 1
        lattice = (packet.coordinates + 1.0) * 0.5 * levels
        if not np.allclose(lattice, np.rint(lattice), atol=1e-9, rtol=0.0):
            raise RuntimeError("coordinate representation is off the declared lattice")
        decoded.append(torch.from_numpy(packet.coordinates).float())
        stream_bytes.append(len(stream))
        hashes.append(hashlib.sha256(stream).hexdigest())
    return (
        torch.stack(decoded).to(coordinates.device),
        np.asarray(stream_bytes, dtype=np.float64),
        tuple(hashes),
    )


def _as_extracted(
    values: Tensor,
    stream_bytes: NDArray[np.float64],
    stream_hashes: tuple[str, ...],
) -> ExtractedBatch:
    rows = tuple(
        row.detach().cpu().numpy().astype(np.float32, copy=True) for row in values
    )
    return ExtractedBatch(
        rows,
        stream_bytes.copy(),
        stream_hashes,
        tuple("" for _ in rows),
    )


class FrozenRepresentationExtractor:
    """Label-free wrapper around all frozen representation encoders.

    ``extract`` intentionally accepts only source points. Category labels,
    model identifiers, normals, and independently resampled targets cannot be
    passed to the learned encoder path by construction.
    """

    def __init__(
        self,
        config: DownstreamClassificationConfig,
        stability: StabilityExperimentConfig,
        *,
        decoder: nn.Module,
        refiner: nn.Module | None,
        feature_codec: VariableFeatureCodec | None,
        device: torch.device,
    ) -> None:
        self.config = config
        self.stability = stability
        self.decoder = decoder
        self.refiner = refiner
        self.feature_codec = feature_codec
        self.device = device

    def extract(self, source_points: Tensor) -> dict[str, ExtractedBatch]:
        """Extract frozen representations using source points and nothing else."""

        if source_points.ndim != 3 or source_points.shape[-1] != 3:
            raise ValueError("source_points must have shape (B, N, 3)")
        if source_points.shape[1] != self.stability.num_points:
            raise ValueError("source point count differs from the frozen protocol")
        source = source_points.to(self.device)
        results: dict[str, ExtractedBatch] = {}
        if self.config.include_source:
            rows = tuple(
                row.detach().cpu().numpy().astype(np.float32, copy=True)
                for row in source
            )
            results["source_full"] = ExtractedBatch(
                rows,
                np.full(len(rows), np.nan, dtype=np.float64),
                tuple("" for _ in rows),
                tuple("" for _ in rows),
            )

        for size in self.config.constellation_sizes:
            with torch.no_grad():
                fps = _fps(source, size, self.stability.coordinate_bits)
            fps, fps_bytes, fps_hashes = _serialize_coordinates(
                fps,
                bits=self.stability.coordinate_bits,
                mode="fps",
                output_points=self.stability.num_points,
            )
            if np.any(
                fps_bytes != expected_stream_bytes(size, self.stability.coordinate_bits)
            ):
                raise RuntimeError("FPS stream has an unexpected complete byte count")
            if self.config.include_fps:
                results[f"fps_raw_k{size:04d}"] = _as_extracted(
                    fps, fps_bytes, fps_hashes
                )
                with torch.no_grad():
                    reconstruction = self.decoder(
                        fps, num_output_points=self.stability.num_points
                    )
                results[f"fps_decoded_k{size:04d}"] = _as_extracted(
                    reconstruction, fps_bytes, fps_hashes
                )

            if self.config.include_adam:
                scorer = _source_scorer(
                    self.decoder,
                    source,
                    num_output_points=self.stability.num_points,
                    chunk_size=self.stability.distance_chunk_size,
                )
                search = _search_adam_start(
                    scorer,
                    fps,
                    bits=self.stability.coordinate_bits,
                    budget=self.config.adam_budget,
                    learning_rate=self.config.adam_learning_rate,
                )
                if search.decoder_evaluations_per_cloud != self.config.adam_budget:
                    raise RuntimeError("Adam encoder did not use its declared budget")
                adam, adam_bytes, adam_hashes = _serialize_coordinates(
                    search.coordinates,
                    bits=self.stability.coordinate_bits,
                    mode="free",
                    output_points=self.stability.num_points,
                )
                results[f"adam{self.config.adam_budget}_raw_k{size:04d}"] = (
                    _as_extracted(adam, adam_bytes, adam_hashes)
                )
                with torch.no_grad():
                    reconstruction = self.decoder(
                        adam, num_output_points=self.stability.num_points
                    )
                results[f"adam{self.config.adam_budget}_decoded_k{size:04d}"] = (
                    _as_extracted(reconstruction, adam_bytes, adam_hashes)
                )

            if (
                self.config.include_refiner
                and size == self.config.primary_constellation_size
            ):
                if self.refiner is None:
                    raise RuntimeError("refiner representation requested without model")
                refined = self.refiner(
                    source,
                    size,
                    decoder=self.decoder,
                    target=source,
                    num_output_points=self.stability.num_points,
                )
                refined, refiner_bytes, refiner_hashes = _serialize_coordinates(
                    refined,
                    bits=self.stability.coordinate_bits,
                    mode="free",
                    output_points=self.stability.num_points,
                )
                results[f"refiner_raw_k{size:04d}"] = _as_extracted(
                    refined, refiner_bytes, refiner_hashes
                )
                with torch.no_grad():
                    reconstruction = self.decoder(
                        refined, num_output_points=self.stability.num_points
                    )
                results[f"refiner_decoded_k{size:04d}"] = _as_extracted(
                    reconstruction, refiner_bytes, refiner_hashes
                )

        if self.config.include_feature:
            if self.feature_codec is None:
                raise RuntimeError("feature representation requested without codec")
            feature_bits = self.feature_codec.encoder.bits
            with torch.no_grad():
                for size, latent_dim in zip(
                    self.config.constellation_sizes,
                    self.config.feature_latent_dims,
                    strict=True,
                ):
                    encoded = self.feature_codec.encoder(source, latent_dim)
                    decoded_rows = []
                    byte_counts = []
                    hashes = []
                    for row in encoded.detach().cpu().numpy():
                        stream = encode_features(
                            row,
                            bits=feature_bits,
                            output_points=self.stability.num_points,
                        )
                        packet = decode_features(stream)
                        repeated = encode_features(
                            packet.features,
                            bits=packet.bits,
                            output_points=packet.output_points,
                        )
                        if repeated != stream:
                            raise RuntimeError(
                                "feature representation failed exact round trip"
                            )
                        decoded_rows.append(packet.features.astype(np.float32))
                        byte_counts.append(len(stream))
                        hashes.append(hashlib.sha256(stream).hexdigest())
                    expected = expected_stream_bytes(
                        size, self.stability.coordinate_bits
                    )
                    if any(value != expected for value in byte_counts):
                        raise RuntimeError("feature representation is not byte-matched")
                    results[f"feature_latent_k{size:04d}"] = ExtractedBatch(
                        tuple(decoded_rows),
                        np.asarray(byte_counts, dtype=np.float64),
                        tuple(hashes),
                        tuple("" for _ in decoded_rows),
                    )
        return results

    def extract_gpcc(
        self,
        source_points: Tensor,
        *,
        work_root: Path,
        sample_offset: int,
    ) -> dict[str, ExtractedBatch]:
        """Run nearest-rate G-PCC using only source points and numeric work IDs."""

        gpcc = self.config.gpcc
        if gpcc is None:
            return {}
        source = source_points.detach().cpu().numpy()
        target_sizes = {
            size: expected_stream_bytes(size, self.stability.coordinate_bits)
            for size in self.config.constellation_sizes
        }
        by_key: dict[str, list[Any]] = {
            f"gpcc_decoded_k{size:04d}": [] for size in self.config.constellation_sizes
        }
        for batch_index, points in enumerate(source):
            candidates = []
            for rate_point in gpcc.rate_points:
                effective = Tmc3RatePoint(
                    rate_point.name,
                    (*gpcc.encoder_args, *rate_point.encoder_args),
                )
                numeric_id = sample_offset + batch_index
                rate_dir = work_root / f"sample_{numeric_id:06d}" / rate_point.name
                result = run_tmc3(
                    Path(gpcc.executable),
                    points,
                    rate_point=effective,
                    work_dir=rate_dir,
                    position_bits=gpcc.position_bits,
                    timeout_seconds=gpcc.timeout_seconds,
                )
                stream_path = rate_dir / "stream.bin"
                stream_hash = file_sha256(stream_path)
                candidates.append((rate_point.name, result, stream_hash))
            for size, target_bytes in target_sizes.items():
                selected = min(
                    candidates,
                    key=lambda item: (
                        abs(item[1].stream_bytes - target_bytes),
                        item[1].stream_bytes,
                        item[0],
                    ),
                )
                by_key[f"gpcc_decoded_k{size:04d}"].append(selected)

        extracted = {}
        for key, selected_rows in by_key.items():
            values = tuple(
                row[1].reconstruction.astype(np.float32, copy=True)
                for row in selected_rows
            )
            extracted[key] = ExtractedBatch(
                values,
                np.asarray(
                    [row[1].stream_bytes for row in selected_rows],
                    dtype=np.float64,
                ),
                tuple(row[2] for row in selected_rows),
                tuple(row[0] for row in selected_rows),
            )
        return extracted


def _padded(
    values: Sequence[FloatArray], input_kind: str
) -> tuple[FloatArray, NDArray[np.int64]]:
    if not values:
        raise ValueError("cannot cache an empty representation")
    lengths = np.asarray([len(row) for row in values], dtype=np.int64)
    if input_kind == "feature":
        if len(set(lengths.tolist())) != 1:
            raise ValueError("feature vectors must have one fixed dimension")
        return np.stack(values).astype(np.float32), lengths
    maximum = int(lengths.max())
    output = np.zeros((len(values), maximum, 3), dtype=np.float32)
    for index, row in enumerate(values):
        if row.ndim != 2 or row.shape[1] != 3:
            raise ValueError("point-set row must have shape (N, 3)")
        output[index, : len(row)] = row
    return output, lengths


def _cache_paths(output_dir: Path, split: str, key: str) -> tuple[Path, Path]:
    cache_dir = output_dir / "representations"
    return cache_dir / f"{split}__{key}.npz", cache_dir / f"{split}__{key}.json"


def _save_cache(
    cache: CachedRepresentation,
    *,
    output_dir: Path,
    split: str,
    fingerprint: str,
    source_points_sha256: str,
) -> None:
    npz_path, metadata_path = _cache_paths(output_dir, split, cache.spec.key)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=npz_path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        np.savez_compressed(
            handle,
            values=cache.values,
            lengths=cache.lengths,
            categories=cache.categories,
            model_ids=cache.model_ids,
            sample_ids=cache.sample_ids,
            stream_bytes=cache.stream_bytes,
            stream_sha256=cache.stream_sha256,
            rate_points=cache.rate_points,
        )
    temporary_path.replace(npz_path)
    metadata = {
        "cache_version": CACHE_VERSION,
        "split": split,
        "spec": asdict(cache.spec),
        "fingerprint": fingerprint,
        "npz_sha256": file_sha256(npz_path),
        "source_points_sha256": source_points_sha256,
        "encoder_inputs": ["source_points"],
        "encoder_forbidden_inputs": [
            "category",
            "label",
            "model_id",
            "normals",
            "target_points",
            "fresh_points",
        ],
        "samples": len(cache.values),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def _load_cache(
    spec: RepresentationSpec,
    *,
    output_dir: Path,
    split: str,
    fingerprint: str,
) -> CachedRepresentation | None:
    npz_path, metadata_path = _cache_paths(output_dir, split, spec.key)
    if not npz_path.exists() and not metadata_path.exists():
        return None
    if not npz_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"partial representation cache for {split}/{spec.key}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "cache_version": CACHE_VERSION,
        "split": split,
        "spec": asdict(spec),
        "fingerprint": fingerprint,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            f"representation cache identity differs for {split}/{spec.key}"
        )
    if file_sha256(npz_path) != metadata.get("npz_sha256"):
        raise RuntimeError(f"representation cache hash failed for {split}/{spec.key}")
    with np.load(npz_path, allow_pickle=False) as values:
        cache = CachedRepresentation(
            spec=spec,
            values=values["values"].astype(np.float32, copy=True),
            lengths=values["lengths"].astype(np.int64, copy=True),
            categories=values["categories"].astype(np.str_, copy=True),
            model_ids=values["model_ids"].astype(np.str_, copy=True),
            sample_ids=values["sample_ids"].astype(np.int64, copy=True),
            stream_bytes=values["stream_bytes"].astype(np.float64, copy=True),
            stream_sha256=values["stream_sha256"].astype(np.str_, copy=True),
            rate_points=values["rate_points"].astype(np.str_, copy=True),
        )
    if len(cache.values) != metadata.get("samples"):
        raise RuntimeError("representation cache sample count differs from metadata")
    return cache


def _source_tensor(sample: Mapping[str, Any]) -> Tensor:
    value = sample.get("source_points", sample.get("points"))
    if not isinstance(value, Tensor):
        raise ValueError("dataset sample is missing source points")
    return value


def _sample_metadata(sample: Mapping[str, Any], fallback: int) -> tuple[str, str, int]:
    def scalar(name: str, default: Any) -> Any:
        value = sample.get(name, default)
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(f"sample metadata {name} must be scalar")
            return value.item()
        return value

    category = str(scalar("category", scalar("family", "unknown")))
    sample_id = int(scalar("sample_id", fallback))
    model_id = str(scalar("model_id", sample_id))
    return category, model_id, sample_id


def _category_for_index(dataset: Dataset[dict[str, Any]], index: int) -> str:
    if isinstance(dataset, Subset):
        parent_index = int(dataset.indices[index])
        return _category_for_index(dataset.dataset, parent_index)  # type: ignore[arg-type]
    if isinstance(dataset, MeshSurfaceDataset):
        return str(dataset.records[index]["category"])
    if isinstance(dataset, ProceduralPointCloudDataset):
        return FAMILIES[index % len(FAMILIES)]
    sample = dataset[index]
    return _sample_metadata(sample, index)[0]


def _selected_indices(
    dataset: Dataset[dict[str, Any]], max_per_category: int | None
) -> tuple[int, ...]:
    if max_per_category is None:
        return tuple(range(len(dataset)))
    counts: dict[str, int] = {}
    selected = []
    for index in range(len(dataset)):
        category = _category_for_index(dataset, index)
        if counts.get(category, 0) >= max_per_category:
            continue
        selected.append(index)
        counts[category] = counts.get(category, 0) + 1
    return tuple(selected)


def _validate_feature_protocol(
    feature: FeatureCodecBenchmarkConfig,
    feature_metrics: Mapping[str, Any],
    stability: StabilityExperimentConfig,
    data_protocol: Mapping[str, Any],
    *,
    manifest_override: str | None,
) -> dict[str, Any]:
    if feature.data_seed != stability.data_seed:
        raise RuntimeError("feature and coordinate codecs use different data seeds")
    if feature.num_points != stability.num_points:
        raise RuntimeError("feature and coordinate codecs use different point counts")
    if feature.coordinate_bits != stability.coordinate_bits:
        raise RuntimeError("feature and coordinate rate protocols use different q")
    if tuple(feature.matched_constellation_sizes) != tuple(
        stability.training_constellation_sizes
    ):
        raise RuntimeError("feature and coordinate rate curricula do not align")

    manifest_path = Path(manifest_override or feature.dataset_manifest)
    manifest = load_mesh_manifest(manifest_path)
    feature_identities = {
        split: {
            f"{record['category']}:{record['model_id']}"
            for record in manifest["splits"][split]
        }
        for split in ("train", "validation", "category_ood")
    }
    coordinate_identities = {
        "train": set(data_protocol["partitions"]["train"]["records"]),
        "validation": set(data_protocol["partitions"]["validation"]["records"]),
        "category_ood": set(data_protocol["partitions"]["ood"]["records"]),
    }
    if feature_identities != coordinate_identities:
        raise RuntimeError(
            "feature checkpoint training/evaluation meshes do not match Experiment 019"
        )
    manifest_hash = file_sha256(manifest_path)
    recorded_hash = feature_metrics.get("protocol", {}).get("manifest_sha256")
    if recorded_hash is not None and recorded_hash != manifest_hash:
        raise RuntimeError("feature artifact manifest hash differs from checked data")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "split_membership_matches_experiment_019": True,
    }


def _load_runtime(
    config: DownstreamClassificationConfig,
    *,
    device: torch.device,
) -> tuple[
    StabilityExperimentConfig,
    FeatureCodecBenchmarkConfig | None,
    dict[str, Dataset[dict[str, Any]]],
    FrozenRepresentationExtractor,
    dict[str, Any],
]:
    stability_path = Path(config.stability_config)
    checked_stability = StabilityExperimentConfig.from_json(stability_path)
    stability_metrics_path = Path(config.stability_artifact_dir) / (
        "stability_metrics.json"
    )
    stability_metrics = json.loads(stability_metrics_path.read_text())
    if stability_metrics.get("config") != _json_ready(asdict(checked_stability)):
        raise RuntimeError("Experiment 019 artifact config differs from checked config")
    contract = stability_metrics.get("contract_checks", {})
    if not contract or not all(bool(value) for value in contract.values()):
        raise RuntimeError("Experiment 019 artifact has a failed scientific contract")
    if config.decoder_seed not in checked_stability.decoder_seeds:
        raise ValueError("selected decoder seed is absent from Experiment 019")
    if config.refiner_seed not in checked_stability.refiner_seeds:
        raise ValueError("selected refiner seed is absent from Experiment 019")
    if config.include_refiner and (
        config.primary_constellation_size != checked_stability.constellation_size
    ):
        raise ValueError("refiner is only declared at Experiment 019's primary K")
    if not set(config.constellation_sizes) <= set(
        checked_stability.training_constellation_sizes
    ):
        raise ValueError("requested K was absent from decoder training curriculum")

    stability = replace(
        checked_stability,
        dataset_root=(config.dataset_root_override or checked_stability.dataset_root),
        dataset_manifest=(
            config.dataset_manifest_override or checked_stability.dataset_manifest
        ),
    )
    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    if data_protocol != stability_metrics.get("data_protocol"):
        raise RuntimeError("Experiment 019 data identity changed before Experiment 030")

    official = OfficialStabilityConfig(
        stability_config=config.stability_config,
        stability_artifact_dir=config.stability_artifact_dir,
        decoder_seeds=checked_stability.decoder_seeds,
        refiner_seeds=checked_stability.refiner_seeds,
        splits=("validation",),
    )
    decoder, refiner, coordinate_model = _load_models(
        stability,
        official,
        decoder_seed=config.decoder_seed,
        refiner_seed=config.refiner_seed if config.include_refiner else None,
        device=device,
    )

    feature_config = None
    feature_codec = None
    feature_identity = None
    if config.include_feature:
        assert config.feature_config is not None
        assert config.feature_artifact_dir is not None
        feature_path = Path(config.feature_config)
        feature_config = FeatureCodecBenchmarkConfig.from_json(feature_path)
        feature_metrics_path = Path(config.feature_artifact_dir) / (
            "multiseed_metrics.json"
        )
        feature_metrics = json.loads(feature_metrics_path.read_text())
        if feature_metrics.get("config") != _json_ready(asdict(feature_config)):
            raise RuntimeError(
                "Experiment 018 artifact config differs from checked config"
            )
        if config.feature_model_seed not in feature_config.model_seeds:
            raise ValueError("selected feature seed is absent from Experiment 018")
        if tuple(config.feature_latent_dims) != tuple(feature_config.latent_dims):
            raise ValueError("configured feature dimensions differ from Experiment 018")
        protocol_identity = _validate_feature_protocol(
            feature_config,
            feature_metrics,
            stability,
            data_protocol,
            manifest_override=config.feature_dataset_manifest_override,
        )
        feature_codec = VariableFeatureCodec(
            feature_config.num_points,
            max(feature_config.latent_dims),
            bits=feature_config.feature_bits,
            feature_width=feature_config.feature_width,
        ).to(device)
        model_dir = (
            Path(config.feature_artifact_dir)
            / f"seed_{config.feature_model_seed}"
            / "model"
        )
        encoder_path = model_dir / "encoder.pt"
        decoder_path = model_dir / "decoder.pt"
        feature_codec.encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=True)
        )
        feature_codec.decoder.load_state_dict(
            torch.load(decoder_path, map_location=device, weights_only=True)
        )
        feature_codec.eval().requires_grad_(False)
        expected_hash = next(
            result["model"]["state_hash"]
            for result in feature_metrics["per_seed"]
            if result["model_seed"] == config.feature_model_seed
        )
        if _state_hash(feature_codec) != expected_hash:
            raise RuntimeError("feature codec checkpoint state hash failed")
        feature_identity = {
            "config_sha256": file_sha256(feature_path),
            "metrics_sha256": file_sha256(feature_metrics_path),
            "encoder_checkpoint_sha256": file_sha256(encoder_path),
            "decoder_checkpoint_sha256": file_sha256(decoder_path),
            "state_hash": expected_hash,
            **protocol_identity,
        }

    official_identity = None
    if config.official_stability_artifact_dir is not None:
        official_path = Path(config.official_stability_artifact_dir) / (
            "official_metrics.json"
        )
        official_metrics = json.loads(official_path.read_text())
        official_contract = official_metrics.get("contract_checks", {})
        if not official_contract or not all(
            bool(value) for value in official_contract.values()
        ):
            raise RuntimeError("Experiment 020 official artifact failed its contract")
        if official_metrics.get("expected_stream_bytes") != expected_stream_bytes(
            checked_stability.constellation_size,
            checked_stability.coordinate_bits,
        ):
            raise RuntimeError("Experiment 020 primary stream size changed")
        official_identity = {
            "metrics_path": str(official_path),
            "metrics_sha256": file_sha256(official_path),
            "contract_checks_pass": True,
        }

    gpcc_identity = None
    if config.gpcc is not None:
        executable = Path(config.gpcc.executable)
        if not executable.is_file() or not executable.stat().st_mode & 0o111:
            raise FileNotFoundError(f"TMC13 is missing or not executable: {executable}")
        gpcc_identity = {
            "executable": str(executable),
            "executable_sha256": file_sha256(executable),
            "position_bits": config.gpcc.position_bits,
        }

    extractor = FrozenRepresentationExtractor(
        config,
        stability,
        decoder=decoder,
        refiner=refiner,
        feature_codec=feature_codec,
        device=device,
    )
    identity = {
        "stability_config_sha256": file_sha256(stability_path),
        "stability_metrics_sha256": file_sha256(stability_metrics_path),
        "data_protocol": data_protocol,
        "coordinate_model": coordinate_model,
        "feature_model": feature_identity,
        "official_reference": official_identity,
        "gpcc": gpcc_identity,
    }
    return stability, feature_config, datasets, extractor, identity


def _clear_cache(
    specs: Sequence[RepresentationSpec], *, output_dir: Path, split: str
) -> None:
    for spec in specs:
        for path in _cache_paths(output_dir, split, spec.key):
            if path.is_file():
                path.unlink()


def _extract_split(
    split: str,
    dataset: Dataset[dict[str, Any]],
    indices: Sequence[int],
    specs: Sequence[RepresentationSpec],
    extractor: FrozenRepresentationExtractor,
    *,
    output_dir: Path,
    fingerprint: str,
    batch_size: int,
) -> dict[str, CachedRepresentation]:
    cached = {
        spec.key: _load_cache(
            spec,
            output_dir=output_dir,
            split=split,
            fingerprint=fingerprint,
        )
        for spec in specs
    }
    if all(value is not None for value in cached.values()):
        return {key: value for key, value in cached.items() if value is not None}

    values: dict[str, list[FloatArray]] = {spec.key: [] for spec in specs}
    byte_counts: dict[str, list[float]] = {spec.key: [] for spec in specs}
    stream_hashes: dict[str, list[str]] = {spec.key: [] for spec in specs}
    rate_points: dict[str, list[str]] = {spec.key: [] for spec in specs}
    categories: list[str] = []
    model_ids: list[str] = []
    sample_ids: list[int] = []
    source_digest = hashlib.sha256()
    work_root = output_dir / "gpcc_work" / split

    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        samples = [dataset[index] for index in batch_indices]
        source = torch.stack([_source_tensor(sample) for sample in samples])
        source_digest.update(source.numpy().tobytes())

        # This is the complete learned encoder call. Labels are deliberately
        # read only below, after the call has returned.
        extracted = extractor.extract(source)
        extracted.update(
            extractor.extract_gpcc(
                source,
                work_root=work_root,
                sample_offset=batch_start,
            )
        )
        if set(extracted) != {spec.key for spec in specs}:
            missing = {spec.key for spec in specs} - set(extracted)
            extra = set(extracted) - {spec.key for spec in specs}
            raise RuntimeError(
                f"representation extraction mismatch; missing={missing}, extra={extra}"
            )
        for spec in specs:
            batch = extracted[spec.key]
            if len(batch.values) != len(samples):
                raise RuntimeError("representation batch does not align with source")
            values[spec.key].extend(batch.values)
            byte_counts[spec.key].extend(batch.stream_bytes.tolist())
            stream_hashes[spec.key].extend(batch.stream_sha256)
            rate_points[spec.key].extend(batch.rate_points)

        for fallback, sample in zip(batch_indices, samples, strict=True):
            category, model_id, sample_id = _sample_metadata(sample, fallback)
            categories.append(category)
            model_ids.append(model_id)
            sample_ids.append(sample_id)

    result: dict[str, CachedRepresentation] = {}
    for spec in specs:
        existing = cached[spec.key]
        if existing is not None:
            result[spec.key] = existing
            continue
        padded, lengths = _padded(values[spec.key], spec.input_kind)
        cache = CachedRepresentation(
            spec=spec,
            values=padded,
            lengths=lengths,
            categories=np.asarray(categories, dtype=np.str_),
            model_ids=np.asarray(model_ids, dtype=np.str_),
            sample_ids=np.asarray(sample_ids, dtype=np.int64),
            stream_bytes=np.asarray(byte_counts[spec.key], dtype=np.float64),
            stream_sha256=np.asarray(stream_hashes[spec.key], dtype=np.str_),
            rate_points=np.asarray(rate_points[spec.key], dtype=np.str_),
        )
        if (
            spec.target_stream_bytes is not None
            and spec.method != "gpcc"
            and np.any(cache.stream_bytes != spec.target_stream_bytes)
        ):
            raise RuntimeError(f"{spec.key} differs from its exact declared bytes")
        _save_cache(
            cache,
            output_dir=output_dir,
            split=split,
            fingerprint=fingerprint,
            source_points_sha256=source_digest.hexdigest(),
        )
        result[spec.key] = cache
    return result


def _assert_cache_alignment(
    caches: Mapping[str, CachedRepresentation], *, split: str
) -> None:
    if not caches:
        raise RuntimeError(f"no representations were cached for split={split}")
    reference = next(iter(caches.values()))
    for cache in caches.values():
        if not (
            np.array_equal(cache.categories, reference.categories)
            and np.array_equal(cache.model_ids, reference.model_ids)
            and np.array_equal(cache.sample_ids, reference.sample_ids)
        ):
            raise RuntimeError(
                f"representation caches are misaligned for split={split}"
            )


def retrieval_average_precision(
    embeddings: NDArray[np.float64] | FloatArray,
    categories: Sequence[str],
    *,
    k: int,
) -> NDArray[np.float64]:
    """Return deterministic within-split AP@k for every eligible query."""

    values = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(categories, dtype=np.str_)
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("embeddings and categories must align")
    if k < 1:
        raise ValueError("retrieval k must be positive")
    distances = np.sum((values[:, None] - values[None, :]) ** 2, axis=2)
    output = np.full(len(values), np.nan, dtype=np.float64)
    indices = np.arange(len(values))
    for query in range(len(values)):
        relevant_total = int(np.count_nonzero(labels == labels[query]) - 1)
        if relevant_total < 1:
            continue
        order = np.lexsort((indices, distances[query]))
        order = order[order != query][:k]
        relevant = labels[order] == labels[query]
        precision = np.cumsum(relevant) / np.arange(1, len(order) + 1)
        denominator = min(relevant_total, k)
        output[query] = float(np.sum(precision * relevant) / denominator)
    return output


def aggregate_accuracy(
    values: NDArray[np.float64] | Sequence[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Aggregate seed or seed-by-cloud scores with a deterministic CI."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim not in {1, 2} or not len(scores):
        raise ValueError("scores must have shape S or S x C")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1")
    if not np.isfinite(scores).any():
        return {
            "mean": None,
            "per_seed": [None for _ in range(len(scores))],
            "confidence_level": confidence_level,
            "confidence_interval_lower": None,
            "confidence_interval_upper": None,
            "bootstrap_samples": samples,
        }

    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    if scores.ndim == 1:
        finite = scores[np.isfinite(scores)]
        for index in range(samples):
            draws[index] = float(
                finite[rng.integers(0, len(finite), size=len(finite))].mean()
            )
        per_seed = [float(value) if np.isfinite(value) else None for value in scores]
    else:
        valid_columns = np.flatnonzero(np.isfinite(scores).any(axis=0))
        if not len(valid_columns):
            raise ValueError("score matrix contains no finite cloud column")
        for index in range(samples):
            seed_draw = rng.integers(0, scores.shape[0], size=scores.shape[0])
            cloud_draw = valid_columns[
                rng.integers(0, len(valid_columns), size=len(valid_columns))
            ]
            sampled = scores[seed_draw[:, None], cloud_draw[None, :]]
            draws[index] = float(np.nanmean(sampled))
        per_seed = [
            float(np.nanmean(row)) if np.isfinite(row).any() else None for row in scores
        ]
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(draws, (alpha, 1.0 - alpha))
    return {
        "mean": float(np.nanmean(scores)),
        "per_seed": per_seed,
        "confidence_level": confidence_level,
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "bootstrap_samples": samples,
    }


def _paired_difference(
    candidate: NDArray[np.float64],
    baseline: NDArray[np.float64],
    *,
    config: DownstreamClassificationConfig,
    seed: int,
) -> dict[str, Any]:
    if candidate.shape != baseline.shape:
        raise ValueError("paired downstream score matrices must have equal shape")
    result = aggregate_accuracy(
        candidate - baseline,
        samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
        seed=seed,
    )
    return {
        "candidate_minus_baseline": result["mean"],
        "per_seed": result["per_seed"],
        "confidence_level": result["confidence_level"],
        "confidence_interval_lower": result["confidence_interval_lower"],
        "confidence_interval_upper": result["confidence_interval_upper"],
        "bootstrap_samples": result["bootstrap_samples"],
        "passes_positive_interval": bool(
            result["confidence_interval_lower"] is not None
            and result["confidence_interval_lower"] > 0.0
        ),
    }


def _classifier_model(
    cache: CachedRepresentation,
    *,
    num_classes: int,
    config: ClassifierConfig,
) -> nn.Module:
    if cache.spec.input_kind == "point_set":
        return SmallPointNet(
            num_classes,
            hidden_width=config.hidden_width,
            embedding_dim=config.embedding_dim,
        )
    return FeatureMLP(
        cache.values.shape[1],
        num_classes,
        hidden_width=config.hidden_width,
        embedding_dim=config.embedding_dim,
    )


def _tensor_batch(
    cache: CachedRepresentation,
    indices: NDArray[np.int64],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    values = torch.from_numpy(cache.values[indices]).to(device)
    lengths = torch.from_numpy(cache.lengths[indices]).to(device)
    return values, lengths


def _evaluate_classifier(
    model: nn.Module,
    cache: CachedRepresentation,
    *,
    category_to_index: Mapping[str, int],
    batch_size: int,
    retrieval_k: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits = []
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(cache.values), batch_size):
            indices = np.arange(
                start, min(start + batch_size, len(cache.values)), dtype=np.int64
            )
            values, lengths = _tensor_batch(cache, indices, device=device)
            batch_logits, batch_embeddings = model(values, lengths)
            logits.append(batch_logits.cpu().numpy())
            embeddings.append(batch_embeddings.cpu().numpy())
    logits_array = np.concatenate(logits)
    embedding_array = np.concatenate(embeddings)
    predictions = logits_array.argmax(axis=1).astype(np.int64)
    known = all(category in category_to_index for category in cache.categories)
    correct = None
    if known:
        labels = np.asarray(
            [category_to_index[str(category)] for category in cache.categories],
            dtype=np.int64,
        )
        correct = (predictions == labels).astype(np.float64)
    retrieval = retrieval_average_precision(
        embedding_array,
        cache.categories.tolist(),
        k=retrieval_k,
    )
    return {
        "predictions": predictions,
        "correct": correct,
        "retrieval_ap": retrieval,
    }


def train_classifier(
    train: CachedRepresentation,
    evaluations: Mapping[str, CachedRepresentation],
    *,
    category_to_index: Mapping[str, int],
    config: ClassifierConfig,
    seed: int,
    retrieval_k: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one deterministic classifier under the common update budget."""

    if set(train.categories) != set(category_to_index):
        raise ValueError("training cache categories differ from the class vocabulary")
    set_seed(seed)
    model = _classifier_model(
        train,
        num_classes=len(category_to_index),
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    labels = np.asarray(
        [category_to_index[str(category)] for category in train.categories],
        dtype=np.int64,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    update_count = 0
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(train.values), generator=generator).numpy()
        total_loss = 0.0
        examples = 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size].astype(np.int64)
            values, lengths = _tensor_batch(train, indices, device=device)
            target = torch.from_numpy(labels[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(values, lengths)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            examples += len(indices)
            update_count += 1
        history.append(
            {
                "epoch": epoch + 1,
                "training_cross_entropy": total_loss / examples,
            }
        )
    split_results = {
        split: _evaluate_classifier(
            model,
            cache,
            category_to_index=category_to_index,
            batch_size=config.batch_size,
            retrieval_k=retrieval_k,
            device=device,
        )
        for split, cache in evaluations.items()
    }
    return model, {
        "seed": seed,
        "history": history,
        "optimizer_updates": update_count,
        "training_elapsed_seconds": time.perf_counter() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_hash": _state_hash(model),
        "splits": split_results,
    }


def _json_classifier_seed(result: Mapping[str, Any]) -> dict[str, Any]:
    splits = {}
    for split, values in result["splits"].items():
        correct = values["correct"]
        retrieval = values["retrieval_ap"]
        splits[split] = {
            "top1_accuracy": (float(np.mean(correct)) if correct is not None else None),
            "predictions": values["predictions"].tolist(),
            "correct": correct.astype(bool).tolist() if correct is not None else None,
            "retrieval_ap_per_query": [
                float(value) if np.isfinite(value) else None for value in retrieval
            ],
            "retrieval_map_at_k": (
                float(np.nanmean(retrieval)) if np.isfinite(retrieval).any() else None
            ),
        }
    return {
        "seed": result["seed"],
        "history": result["history"],
        "optimizer_updates": result["optimizer_updates"],
        "training_elapsed_seconds": result["training_elapsed_seconds"],
        "parameters": result["parameters"],
        "state_hash": result["state_hash"],
        "splits": splits,
    }


def _rate_summary(cache: CachedRepresentation) -> dict[str, Any]:
    finite = cache.stream_bytes[np.isfinite(cache.stream_bytes)]
    return {
        "target_stream_bytes": cache.spec.target_stream_bytes,
        "mean_actual_stream_bytes": float(finite.mean()) if len(finite) else None,
        "minimum_actual_stream_bytes": float(finite.min()) if len(finite) else None,
        "maximum_actual_stream_bytes": float(finite.max()) if len(finite) else None,
        "nearest_rate_points": {
            str(name): int(np.count_nonzero(cache.rate_points == name))
            for name in sorted(set(cache.rate_points) - {""})
        },
    }


def _primary_gate(
    score_arrays: Mapping[str, Mapping[str, NDArray[np.float64]]],
    config: DownstreamClassificationConfig,
    *,
    primary_stream_bytes: int,
) -> dict[str, Any]:
    size = config.primary_constellation_size
    candidate = f"adam{config.adam_budget}_raw_k{size:04d}"
    baselines = (f"fps_raw_k{size:04d}", f"feature_latent_k{size:04d}")
    required = {candidate, *baselines}
    if not required <= set(score_arrays):
        return {
            "name": "G-B1",
            "status": "incomplete",
            "missing_representations": sorted(required - set(score_arrays)),
            "passes": False,
        }
    comparisons = []
    comparison_index = 0
    for baseline in baselines:
        for metric in ("validation_top1", "ood_retrieval_map"):
            candidate_values = score_arrays[candidate][metric]
            baseline_values = score_arrays[baseline][metric]
            comparison = _paired_difference(
                candidate_values,
                baseline_values,
                config=config,
                seed=config.bootstrap_seed + 1000 + comparison_index,
            )
            comparisons.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    **comparison,
                }
            )
            comparison_index += 1
    complete = all(row["confidence_interval_lower"] is not None for row in comparisons)
    return {
        "name": "G-B1",
        "status": "complete" if complete else "insufficient_retrieval_relevance",
        "primary_stream_bytes": primary_stream_bytes,
        "definition": (
            f"the raw Adam-{config.adam_budget} constellation must beat the "
            "byte-matched raw FPS subset and feature latent on seen-category "
            "validation top-1 and category-OOD within-split retrieval mAP@k; "
            "every paired classifier-seed/cloud bootstrap interval must exclude zero"
        ),
        "comparisons": comparisons,
        "passes": bool(
            complete and all(row["passes_positive_interval"] for row in comparisons)
        ),
    }


def run_downstream_classification(
    config: DownstreamClassificationConfig,
    *,
    device_name: str = "auto",
    rebuild_cache: bool = False,
) -> dict[str, Any]:
    """Extract frozen inputs, train classifiers, and write Experiment 030 JSON."""

    device = select_device(device_name)
    started = time.perf_counter()
    stability, feature, datasets, extractor, identity = _load_runtime(
        config, device=device
    )
    specs = _representation_specs(config, stability, feature)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_indices = {
        split: _selected_indices(datasets[split], config.max_samples_per_category)
        for split in SPLITS
    }
    fingerprint_payload = {
        "cache_version": CACHE_VERSION,
        "implementation_sha256": file_sha256(Path(__file__)),
        "config": _json_ready(asdict(config)),
        "specs": [asdict(spec) for spec in specs],
        "artifact_identity": identity,
        "selected_indices": {
            split: list(indices) for split, indices in split_indices.items()
        },
    }
    fingerprint = _sha256_json(fingerprint_payload)
    manifest = {
        "experiment": "030_downstream_classification",
        "fingerprint": fingerprint,
        **fingerprint_payload,
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("existing Experiment 030 run manifest differs")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if rebuild_cache:
        for split in SPLITS:
            _clear_cache(specs, output_dir=output_dir, split=split)

    caches = {
        split: _extract_split(
            split,
            datasets[split],
            split_indices[split],
            specs,
            extractor,
            output_dir=output_dir,
            fingerprint=fingerprint,
            batch_size=config.extraction_batch_size,
        )
        for split in SPLITS
    }
    for split in SPLITS:
        _assert_cache_alignment(caches[split], split=split)

    train_reference = next(iter(caches["train"].values()))
    validation_reference = next(iter(caches["validation"].values()))
    ood_reference = next(iter(caches["ood"].values()))
    training_categories = sorted(set(train_reference.categories.tolist()))
    validation_categories = set(validation_reference.categories.tolist())
    ood_categories = set(ood_reference.categories.tolist())
    if not validation_categories <= set(training_categories):
        raise RuntimeError("validation includes categories absent from training")
    if set(training_categories) & ood_categories:
        raise RuntimeError("category-OOD includes a classifier training category")
    category_to_index = {
        category: index for index, category in enumerate(training_categories)
    }

    decoder_hash_after = _state_hash(extractor.decoder)
    if decoder_hash_after != identity["coordinate_model"]["decoder_state_hash"]:
        raise RuntimeError("frozen decoder changed during representation extraction")
    if extractor.refiner is not None and _state_hash(extractor.refiner) != identity[
        "coordinate_model"
    ].get("refiner_state_hash"):
        raise RuntimeError("frozen refiner changed during representation extraction")
    if (
        extractor.feature_codec is not None
        and _state_hash(extractor.feature_codec)
        != identity["feature_model"]["state_hash"]
    ):
        raise RuntimeError(
            "frozen feature codec changed during representation extraction"
        )

    classifier_root = output_dir / "classifiers"
    representation_results = []
    score_arrays: dict[str, dict[str, NDArray[np.float64]]] = {}
    expected_updates = config.classifier_epochs * math.ceil(
        len(train_reference.values) / config.classifier_batch_size
    )
    for representation_index, spec in enumerate(specs):
        train_cache = caches["train"][spec.key]
        evaluation_caches = {
            "validation": caches["validation"][spec.key],
            "ood": caches["ood"][spec.key],
        }
        seed_results = []
        raw_seed_results = []
        for classifier_seed in config.classifier_seeds:
            model, seed_result = train_classifier(
                train_cache,
                evaluation_caches,
                category_to_index=category_to_index,
                config=config.classifier,
                seed=classifier_seed,
                retrieval_k=config.retrieval_k,
                device=device,
            )
            if seed_result["optimizer_updates"] != expected_updates:
                raise RuntimeError(
                    "classifier arm did not use the common update budget"
                )
            checkpoint_dir = classifier_root / spec.key
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"seed_{classifier_seed}.pt"
            torch.save(
                {
                    "experiment": "030_downstream_classification",
                    "spec": asdict(spec),
                    "classifier_seed": classifier_seed,
                    "classifier_config": asdict(config.classifier),
                    "category_to_index": category_to_index,
                    "model": model.state_dict(),
                    "state_hash": seed_result["state_hash"],
                    "run_fingerprint": fingerprint,
                },
                checkpoint_path,
            )
            json_seed = _json_classifier_seed(seed_result)
            json_seed["checkpoint"] = str(checkpoint_path)
            json_seed["checkpoint_sha256"] = file_sha256(checkpoint_path)
            seed_results.append(json_seed)
            raw_seed_results.append(seed_result)

        validation_correct = np.stack(
            [result["splits"]["validation"]["correct"] for result in raw_seed_results]
        )
        validation_retrieval = np.stack(
            [
                result["splits"]["validation"]["retrieval_ap"]
                for result in raw_seed_results
            ]
        )
        ood_retrieval = np.stack(
            [result["splits"]["ood"]["retrieval_ap"] for result in raw_seed_results]
        )
        score_arrays[spec.key] = {
            "validation_top1": validation_correct,
            "validation_retrieval_map": validation_retrieval,
            "ood_retrieval_map": ood_retrieval,
        }
        metric_seed = config.bootstrap_seed + 100 * representation_index
        metrics = {
            "validation": {
                "top1_accuracy": aggregate_accuracy(
                    validation_correct,
                    samples=config.bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=metric_seed,
                ),
                "retrieval_map_at_k": aggregate_accuracy(
                    validation_retrieval,
                    samples=config.bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=metric_seed + 1,
                ),
            },
            "category_ood": {
                "top1_accuracy": None,
                "top1_reason": "zero-shot labels are absent from classifier training",
                "retrieval_map_at_k": aggregate_accuracy(
                    ood_retrieval,
                    samples=config.bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=metric_seed + 2,
                ),
            },
        }
        representation_results.append(
            {
                "spec": asdict(spec),
                "rate": _rate_summary(caches["validation"][spec.key]),
                "classifier_architecture": (
                    "small_pointnet"
                    if spec.input_kind == "point_set"
                    else "feature_mlp"
                ),
                "classifier_seeds": seed_results,
                "metrics": metrics,
            }
        )

    gate = _primary_gate(
        score_arrays,
        config,
        primary_stream_bytes=expected_stream_bytes(
            config.primary_constellation_size,
            stability.coordinate_bits,
        ),
    )
    rate_accuracy_curve = [
        {
            "representation": row["spec"]["key"],
            "method": row["spec"]["method"],
            "view": row["spec"]["view"],
            **row["rate"],
            "validation_top1_accuracy": row["metrics"]["validation"]["top1_accuracy"],
            "validation_retrieval_map_at_k": row["metrics"]["validation"][
                "retrieval_map_at_k"
            ],
            "category_ood_retrieval_map_at_k": row["metrics"]["category_ood"][
                "retrieval_map_at_k"
            ],
        }
        for row in representation_results
    ]
    rate_accuracy_curve.sort(
        key=lambda row: (
            row["target_stream_bytes"] is None,
            row["target_stream_bytes"] or math.inf,
            row["representation"],
        )
    )
    result = {
        "experiment": "030_downstream_classification",
        "config": _json_ready(asdict(config)),
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "protocol": {
            "task": "ModelNet40 seen-category classification and retrieval",
            "classifier_label_vocabulary": training_categories,
            "training_category_count": len(training_categories),
            "validation_category_count": len(validation_categories),
            "category_ood_category_count": len(ood_categories),
            "selected_samples": {
                split: len(indices) for split, indices in split_indices.items()
            },
            "classifier_budget": asdict(config.classifier),
            "classifier_optimizer_updates_per_seed_and_arm": expected_updates,
            "retrieval_metric": (
                f"within-split Euclidean embedding mAP@{config.retrieval_k}; "
                "self matches excluded"
            ),
            "accuracy_ci": (
                "hierarchical classifier-seed and paired-cloud percentile bootstrap"
            ),
            "shared_model_size_excluded_from_per_cloud_rate": True,
            "gpcc_rate_selection": (
                "per-cloud nearest actual serialized bytes without interpolation"
                if config.gpcc is not None
                else None
            ),
        },
        "artifact_identity": identity,
        "contract_checks": {
            "encoder_inputs_are_source_points_only": True,
            "labels_attached_only_after_representation_extraction": True,
            "classifier_trained_after_frozen_representation_cache": True,
            "validation_categories_seen_during_classifier_training": True,
            "category_ood_categories_absent_from_classifier_training": True,
            "frozen_decoder_hash_unchanged": True,
            "frozen_refiner_hash_unchanged": extractor.refiner is None
            or _state_hash(extractor.refiner)
            == identity["coordinate_model"].get("refiner_state_hash"),
            "frozen_feature_codec_hash_unchanged": extractor.feature_codec is None
            or _state_hash(extractor.feature_codec)
            == identity["feature_model"]["state_hash"],
            "common_classifier_update_budget": all(
                seed["optimizer_updates"] == expected_updates
                for row in representation_results
                for seed in row["classifier_seeds"]
            ),
            "exact_non_gpcc_rate_bytes": all(
                row["spec"]["method"] in {"gpcc", "source"}
                or row["rate"]["minimum_actual_stream_bytes"]
                == row["rate"]["target_stream_bytes"]
                == row["rate"]["maximum_actual_stream_bytes"]
                for row in representation_results
            ),
        },
        "representations": representation_results,
        "rate_accuracy_curve": rate_accuracy_curve,
        "primary_gate": gate,
        "elapsed_seconds": time.perf_counter() - started,
        "run_manifest": str(manifest_path),
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError("Experiment 030 scientific contract failed")
    metrics_path = output_dir / "downstream_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_030_downstream_classification_smoke.json"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--stability-config", type=Path)
    parser.add_argument("--stability-artifact-dir", type=Path)
    parser.add_argument("--official-stability-artifact-dir", type=Path)
    parser.add_argument("--feature-config", type=Path)
    parser.add_argument("--feature-artifact-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--feature-dataset-manifest", type=Path)
    parser.add_argument("--tmc3", type=Path)
    parser.add_argument("--skip-gpcc", action="store_true")
    parser.add_argument("--max-samples-per-category", type=int)
    parser.add_argument("--adam-budget", type=int)
    parser.add_argument("--classifier-epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    config = DownstreamClassificationConfig.from_json(args.config)
    replacements = {
        "stability_config": args.stability_config,
        "stability_artifact_dir": args.stability_artifact_dir,
        "official_stability_artifact_dir": args.official_stability_artifact_dir,
        "feature_config": args.feature_config,
        "feature_artifact_dir": args.feature_artifact_dir,
        "dataset_root_override": args.dataset_root,
        "dataset_manifest_override": args.dataset_manifest,
        "feature_dataset_manifest_override": args.feature_dataset_manifest,
        "max_samples_per_category": args.max_samples_per_category,
        "adam_budget": args.adam_budget,
        "classifier_epochs": args.classifier_epochs,
        "output_dir": args.output_dir,
    }
    for name, value in replacements.items():
        if value is not None:
            config = replace(
                config, **{name: str(value) if isinstance(value, Path) else value}
            )
    if args.skip_gpcc:
        config = replace(config, gpcc=None)
    elif args.tmc3 is not None:
        if config.gpcc is None:
            raise ValueError("--tmc3 requires a config with a gpcc section")
        config = replace(
            config,
            gpcc=replace(config.gpcc, executable=str(args.tmc3)),
        )
    result = run_downstream_classification(
        config,
        device_name=args.device,
        rebuild_cache=args.rebuild_cache,
    )
    print(
        json.dumps(
            {
                "primary_gate": result["primary_gate"],
                "representations": len(result["representations"]),
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "downstream_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
