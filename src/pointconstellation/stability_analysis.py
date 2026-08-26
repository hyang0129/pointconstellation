"""Experiment 032: constellation repeatability and cross-decoder stability.

The runner reuses the sealed Experiment 019 decoders/refiners and the
Experiment 021/022 source-only encoders.  Every tested message is an actual
fixed-width bitstream containing only an unordered, quantized ``K x 3``
coordinate set.  Independent surface samples and transformed fresh samples are
used only for evaluation after an encoding has been selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from pointconstellation.bitstream import (
    decode_constellation,
    encode_constellation,
    expected_stream_bytes,
)
from pointconstellation.data import file_sha256
from pointconstellation.headroom_experiment import (
    _search_adam_start,
    _source_scorer,
)
from pointconstellation.official_stability import (
    OfficialStabilityConfig,
    _load_models,
    _synchronize,
)
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.selection_baselines import (
    SELECTION_METHODS,
    SELECTION_REPRESENTATIONS,
)
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _per_cloud_chamfer,
)
from pointconstellation.train import select_device

FloatArray = NDArray[np.float64]

METHODS = ("fps", "random_best_of_16", "refiner", "adam_64")
SPLITS = ("validation", "ood")
COMPARISON_CONDITIONS = (
    "independent_sample",
    "translation",
    "rotation",
    "rotation_pca",
)


def _points(value: ArrayLike, *, name: str) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{name} must have shape (N, 3) with N > 0")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return points


def _nearest_distances(first: FloatArray, second: FloatArray) -> FloatArray:
    squared = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def constellation_matching_metrics(
    first: ArrayLike,
    second: ArrayLike,
    *,
    coordinate_bits: int,
    radii_bins: Sequence[int],
) -> dict[str, Any]:
    """Return unordered symmetric repeatability and Hausdorff diagnostics.

    A point is repeated when its Euclidean nearest-neighbor distance is at most
    ``r`` times the declared coordinate-lattice step.  Repeatability is the
    average of the two directed fractions, so the definition also remains
    explicit for ablations whose cardinalities differ.
    """

    if not 2 <= coordinate_bits <= 24:
        raise ValueError("coordinate_bits must be between 2 and 24")
    radii = tuple(int(radius) for radius in radii_bins)
    if not radii or min(radii) < 0 or len(set(radii)) != len(radii):
        raise ValueError("radii_bins must contain unique nonnegative integers")
    first_points = _points(first, name="first constellation")
    second_points = _points(second, name="second constellation")
    forward = _nearest_distances(first_points, second_points)
    backward = _nearest_distances(second_points, first_points)
    step = 2.0 / ((1 << coordinate_bits) - 1)
    repeatability = {}
    directed = {}
    for radius in radii:
        threshold = radius * step
        first_fraction = float(np.mean(forward <= threshold + 1e-12))
        second_fraction = float(np.mean(backward <= threshold + 1e-12))
        repeatability[str(radius)] = 0.5 * (first_fraction + second_fraction)
        directed[str(radius)] = {
            "first_to_second": first_fraction,
            "second_to_first": second_fraction,
        }
    hausdorff = float(max(forward.max(), backward.max()))
    return {
        "coordinate_bits": coordinate_bits,
        "lattice_step": step,
        "repeatability_by_radius_bins": repeatability,
        "directed_repeatability_by_radius_bins": directed,
        "hausdorff": hausdorff,
        "hausdorff_lattice_bins": hausdorff / step,
    }


def constellation_repeatability(
    first: ArrayLike,
    second: ArrayLike,
    *,
    coordinate_bits: int,
    radius_bins: int,
) -> float:
    """Return the symmetric match fraction at one lattice-bin radius."""

    result = constellation_matching_metrics(
        first,
        second,
        coordinate_bits=coordinate_bits,
        radii_bins=(radius_bins,),
    )
    return float(result["repeatability_by_radius_bins"][str(radius_bins)])


@dataclass(frozen=True)
class RigidTransform:
    """Known row-vector rigid transform ``x R^T + t``."""

    rotation: FloatArray
    translation: FloatArray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(
                "rotation and translation must have shapes (3, 3) and (3,)"
            )
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("rigid transform must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
            raise ValueError("rotation must have determinant one")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> RigidTransform:
        return cls(np.eye(3), np.zeros(3))

    def apply(self, points: ArrayLike) -> FloatArray:
        values = _points(points, name="points")
        return values @ self.rotation.T + self.translation

    def inverse(self, points: ArrayLike) -> FloatArray:
        values = _points(points, name="points")
        return (values - self.translation) @ self.rotation

    def to_json(self) -> dict[str, Any]:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }


def _random_rotation(rng: np.random.Generator) -> FloatArray:
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def random_rigid_transform(
    points: ArrayLike,
    *,
    seed: int,
    maximum_translation: float,
    rotate: bool = True,
    translate: bool = True,
) -> RigidTransform:
    """Draw a deterministic transform that keeps the supplied cloud in-domain."""

    if not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if not 0.0 <= maximum_translation <= 1.0:
        raise ValueError("maximum_translation must be in [0, 1]")
    values = _points(points, name="points")
    rng = np.random.default_rng(seed)
    rotation = _random_rotation(rng) if rotate else np.eye(3)
    rotated = values @ rotation.T
    translation = np.zeros(3)
    if translate:
        requested = rng.uniform(-maximum_translation, maximum_translation, size=3)
        lower = -1.0 - rotated.min(axis=0)
        upper = 1.0 - rotated.max(axis=0)
        translation = np.clip(requested, lower, upper)
    transform = RigidTransform(rotation, translation)
    transformed = transform.apply(values)
    if np.any(transformed < -1.0 - 1e-7) or np.any(transformed > 1.0 + 1e-7):
        raise RuntimeError("generated rigid transform left the coordinate domain")
    return transform


@dataclass(frozen=True)
class PcaCanonicalFrame:
    """Permutation-invariant PCA pose normalization used only as an ablation."""

    center: FloatArray
    axes: FloatArray
    scale: float
    eigenvalues: FloatArray

    def apply(self, points: ArrayLike) -> FloatArray:
        values = _points(points, name="points")
        return ((values - self.center) @ self.axes) / self.scale

    def inverse(self, points: ArrayLike) -> FloatArray:
        values = _points(points, name="points")
        return (values * self.scale) @ self.axes.T + self.center

    def to_json(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "axes": self.axes.tolist(),
            "scale": self.scale,
            "eigenvalues": self.eigenvalues.tolist(),
        }


def pca_canonical_frame(points: ArrayLike) -> PcaCanonicalFrame:
    """Fit a deterministic sign-oriented, right-handed PCA frame."""

    values = _points(points, name="points")
    center = values.mean(axis=0)
    centered = values - center
    covariance = centered.T @ centered / len(centered)
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = axes[:, order]
    for index in range(2):
        projection = centered @ axes[:, index]
        orientation = float(np.sum(projection**3))
        if abs(orientation) <= 1e-14:
            orientation = float(projection[np.argmax(np.abs(projection))])
        if orientation < 0:
            axes[:, index] *= -1.0
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    axes[:, 2] /= np.linalg.norm(axes[:, 2])
    canonical = centered @ axes
    scale = max(1.0, float(np.abs(canonical).max()))
    return PcaCanonicalFrame(center, axes, scale, eigenvalues)


@dataclass(frozen=True)
class ConstellationStabilityConfig:
    """Validated configuration for Experiment 032."""

    stability_config: str = "configs/experiment_019_stability_modelnet40.json"
    stability_artifact_dir: str = "artifacts/local/experiment_019_stability_modelnet40"
    decoder_seeds: tuple[int, ...] = (7, 17, 29, 41, 53, 67)
    refiner_seeds: tuple[int, ...] = (101, 211, 307)
    methods: tuple[str, ...] = METHODS
    splits: tuple[str, ...] = SPLITS
    max_clouds_per_split: int | None = None
    repeatability_radii_bins: tuple[int, ...] = (1, 4, 16, 64)
    transform_seed: int = 20_260_832
    maximum_translation: float = 0.1
    selection_seed: int = 20_260_832
    adam_evaluations: int = 64
    adam_learning_rate: float = 0.03
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_832
    confidence_level: float = 0.95
    gate_radius_bins: int = 16
    gate_minimum_repeatability: float = 0.5
    gate_minimum_translation_repeatability: float = 0.75
    gate_maximum_cross_decoder_rmse_degradation_percent: float = 50.0
    output_dir: str = "artifacts/local/experiment_032_constellation_stability"

    def __post_init__(self) -> None:
        if len(self.decoder_seeds) < 2 or len(set(self.decoder_seeds)) != len(
            self.decoder_seeds
        ):
            raise ValueError("decoder_seeds must contain at least two unique seeds")
        if not self.refiner_seeds or len(set(self.refiner_seeds)) != len(
            self.refiner_seeds
        ):
            raise ValueError("refiner_seeds must be nonempty and unique")
        if tuple(self.methods) != METHODS:
            raise ValueError(f"methods must be exactly {METHODS}")
        if not self.splits or len(set(self.splits)) != len(self.splits):
            raise ValueError("splits must be nonempty and unique")
        if set(self.splits) - set(SPLITS):
            raise ValueError("splits must contain validation and/or ood")
        if self.max_clouds_per_split is not None and self.max_clouds_per_split < 1:
            raise ValueError("max_clouds_per_split must be positive")
        radii = self.repeatability_radii_bins
        if (
            not radii
            or len(set(radii)) != len(radii)
            or min(radii) < 0
            or tuple(sorted(radii)) != radii
        ):
            raise ValueError("repeatability radii must be unique and increasing")
        if self.gate_radius_bins not in radii:
            raise ValueError("gate_radius_bins must be a declared repeatability radius")
        for seed in (self.transform_seed, self.selection_seed, self.bootstrap_seed):
            if not 0 <= seed < 2**63:
                raise ValueError("random seeds must be nonnegative 63-bit integers")
        if not 0 < self.maximum_translation <= 1.0:
            raise ValueError("maximum_translation must be in (0, 1]")
        if self.adam_evaluations < 1 or self.adam_learning_rate <= 0:
            raise ValueError("Adam evaluations and learning rate must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")
        if not 0.0 <= self.gate_minimum_repeatability <= 1.0:
            raise ValueError("gate_minimum_repeatability must be in [0, 1]")
        if not 0.0 <= self.gate_minimum_translation_repeatability <= 1.0:
            raise ValueError("gate translation repeatability must be in [0, 1]")
        if self.gate_maximum_cross_decoder_rmse_degradation_percent < 0:
            raise ValueError("cross-decoder degradation threshold cannot be negative")

    @classmethod
    def from_json(cls, path: Path) -> ConstellationStabilityConfig:
        values = json.loads(path.read_text())
        for key in (
            "decoder_seeds",
            "refiner_seeds",
            "methods",
            "splits",
            "repeatability_radii_bins",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class DecodedQuality:
    """Reconstruction and source/fresh distortion from one coordinate stream."""

    reconstruction: Tensor
    source_chamfer_mse: float
    fresh_chamfer_mse: float


def _as_batched_tensor(points: ArrayLike | Tensor, device: torch.device) -> Tensor:
    if isinstance(points, Tensor):
        tensor = points.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(
            _points(points, name="evaluation points"),
            dtype=torch.float32,
            device=device,
        )
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != 3:
        raise ValueError("evaluation points must have shape (N, 3) or (1, N, 3)")
    return tensor


def evaluate_cross_decoder(
    decoder: nn.Module,
    stream: bytes,
    *,
    source_points: ArrayLike | Tensor,
    fresh_points: ArrayLike | Tensor | None = None,
    device: torch.device | None = None,
    distance_chunk_size: int = 256,
) -> DecodedQuality:
    """Decode one coordinate-only stream with a supplied fixture or real decoder."""

    if distance_chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive")
    if device is None:
        parameter = next(decoder.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
    packet = decode_constellation(stream)
    coordinates = torch.from_numpy(packet.coordinates).float().unsqueeze(0).to(device)
    source = _as_batched_tensor(source_points, device)
    fresh = source if fresh_points is None else _as_batched_tensor(fresh_points, device)
    with torch.no_grad():
        reconstruction = decoder(
            coordinates,
            num_output_points=packet.output_points,
        )
        source_loss = _per_cloud_chamfer(
            reconstruction,
            source,
            chunk_size=distance_chunk_size,
        )
        fresh_loss = _per_cloud_chamfer(
            reconstruction,
            fresh,
            chunk_size=distance_chunk_size,
        )
    return DecodedQuality(
        reconstruction.detach().cpu(),
        float(source_loss.item()),
        float(fresh_loss.item()),
    )


@dataclass(frozen=True)
class _Variant:
    name: str
    source: Tensor
    fresh: Tensor


@dataclass(frozen=True)
class _Encoding:
    coordinates: FloatArray
    stream: bytes
    quality: DecodedQuality
    encode_seconds: float
    representation_class: str
    bitstream_mode: str
    selection_seed: int | None
    decoder_evaluations: int


def _stable_seed(*parts: Any) -> int:
    identity = json.dumps(parts, separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (
        2**63
    )


def _sample_metadata(sample: Mapping[str, Any], fallback: int) -> dict[str, Any]:
    def value(name: str, default: Any) -> Any:
        result = sample.get(name, default)
        if isinstance(result, Tensor):
            if result.numel() != 1:
                raise ValueError(f"sample metadata {name} must be scalar")
            return result.item()
        return result

    sample_id = int(value("sample_id", fallback))
    return {
        "family": str(value("family", value("category", "unknown"))),
        "model_id": str(value("model_id", sample_id)),
        "sample_id": sample_id,
    }


def _sample_tensor(sample: Mapping[str, Any], name: str) -> Tensor:
    value = sample.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"sample is missing tensor {name}")
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"sample tensor {name} must have shape (N, 3)")
    return value.detach().float()


def _variant_inputs(
    sample: Mapping[str, Any],
    *,
    split: str,
    metadata: Mapping[str, Any],
    config: ConstellationStabilityConfig,
    device: torch.device,
) -> tuple[dict[str, _Variant], dict[str, Any]]:
    source = _sample_tensor(sample, "source_points").cpu().numpy().astype(np.float64)
    fresh = _sample_tensor(sample, "fresh_points").cpu().numpy().astype(np.float64)
    if np.array_equal(source, fresh):
        raise RuntimeError("Experiment 032 requires two independent mesh samples")
    identity = (split, metadata["family"], metadata["model_id"], metadata["sample_id"])
    combined = np.concatenate((source, fresh), axis=0)
    translation = random_rigid_transform(
        combined,
        seed=_stable_seed(config.transform_seed, "translation", *identity),
        maximum_translation=config.maximum_translation,
        rotate=False,
        translate=True,
    )
    rotation = random_rigid_transform(
        combined,
        seed=_stable_seed(config.transform_seed, "rotation", *identity),
        maximum_translation=0.0,
        rotate=True,
        translate=False,
    )
    rotated_source = rotation.apply(source)
    rotated_fresh = rotation.apply(fresh)
    pca_source = pca_canonical_frame(source)
    pca_rotation = pca_canonical_frame(rotated_source)

    def variant(name: str, first: FloatArray, second: FloatArray) -> _Variant:
        return _Variant(
            name,
            torch.from_numpy(first).float().to(device),
            torch.from_numpy(second).float().to(device),
        )

    variants = {
        "source": variant("source", source, fresh),
        "independent_sample": variant("independent_sample", fresh, source),
        "translation": variant(
            "translation", translation.apply(source), translation.apply(fresh)
        ),
        "rotation": variant("rotation", rotated_source, rotated_fresh),
        "pca_source": variant(
            "pca_source", pca_source.apply(source), pca_source.apply(fresh)
        ),
        "rotation_pca": variant(
            "rotation_pca",
            pca_rotation.apply(rotated_source),
            pca_rotation.apply(rotated_fresh),
        ),
    }
    bookkeeping = {
        "translation": translation,
        "rotation": rotation,
        "pca_source": pca_source,
        "pca_rotation": pca_rotation,
        "translation_inverse_max_error": float(
            np.abs(translation.inverse(translation.apply(source)) - source).max()
        ),
        "rotation_inverse_max_error": float(
            np.abs(rotation.inverse(rotation.apply(source)) - source).max()
        ),
    }
    return variants, bookkeeping


def _encode(
    variant: _Variant,
    *,
    method: str,
    decoder: nn.Module,
    refiner: nn.Module | None,
    stability: StabilityExperimentConfig,
    config: ConstellationStabilityConfig,
    seed_parts: Sequence[Any],
    device: torch.device,
) -> _Encoding:
    source = variant.source.unsqueeze(0)
    selection_seed: int | None = None
    decoder_evaluations = 0
    representation_class = "free-coordinate"
    bitstream_mode = "free"

    def scorer(candidate: Tensor) -> float:
        with torch.no_grad():
            reconstruction = decoder(
                candidate.unsqueeze(0),
                num_output_points=stability.num_points,
            )
            loss = _per_cloud_chamfer(
                reconstruction,
                source,
                chunk_size=stability.distance_chunk_size,
            )
        return float(loss.item())

    _synchronize(device)
    started = time.perf_counter()
    if method in {"fps", "random_best_of_16"}:
        selection_seed = _stable_seed(config.selection_seed, method, *seed_parts)
        coordinates = SELECTION_METHODS[method](
            variant.source,
            stability.constellation_size,
            stability.coordinate_bits,
            selection_seed,
            scorer if method == "random_best_of_16" else None,
        ).unsqueeze(0)
        representation_class = SELECTION_REPRESENTATIONS[method]
        bitstream_mode = "fps" if method == "fps" else "strict_subset"
        decoder_evaluations = 0 if method == "fps" else 16
    elif method == "refiner":
        if refiner is None:
            raise ValueError("refiner method requires a paired refiner")
        coordinates = refiner(
            source,
            stability.constellation_size,
            decoder=decoder,
            target=source,
            num_output_points=stability.num_points,
        )
        decoder_evaluations = (
            stability.recurrent_steps if stability.use_decoder_gradient else 0
        )
    elif method == "adam_64":
        initial = SELECTION_METHODS["fps"](
            variant.source,
            stability.constellation_size,
            stability.coordinate_bits,
            0,
            None,
        ).unsqueeze(0)
        score = _source_scorer(
            decoder,
            source,
            num_output_points=stability.num_points,
            chunk_size=stability.distance_chunk_size,
        )
        result = _search_adam_start(
            score,
            initial,
            bits=stability.coordinate_bits,
            budget=config.adam_evaluations,
            learning_rate=config.adam_learning_rate,
        )
        coordinates = result.coordinates
        decoder_evaluations = result.decoder_evaluations_per_cloud
    else:
        raise ValueError(f"unknown Experiment 032 method: {method}")

    stream = encode_constellation(
        coordinates[0].detach().cpu().numpy(),
        bits=stability.coordinate_bits,
        mode=bitstream_mode,
        output_points=stability.num_points,
    )
    packet = decode_constellation(stream)
    repeated = encode_constellation(
        packet.coordinates,
        bits=packet.bits,
        mode=packet.mode,
        output_points=packet.output_points,
    )
    lattice = (packet.coordinates + 1.0) * 0.5 * ((1 << packet.bits) - 1)
    if repeated != stream or not np.allclose(lattice, np.rint(lattice), atol=1e-9):
        raise RuntimeError("constellation stream failed exact round-trip checks")
    _synchronize(device)
    encode_seconds = time.perf_counter() - started
    quality = evaluate_cross_decoder(
        decoder,
        stream,
        source_points=variant.source,
        fresh_points=variant.fresh,
        device=device,
        distance_chunk_size=stability.distance_chunk_size,
    )
    return _Encoding(
        packet.coordinates,
        stream,
        quality,
        encode_seconds,
        representation_class,
        bitstream_mode,
        selection_seed,
        decoder_evaluations,
    )


def _chamfer_rmse(first: ArrayLike, second: ArrayLike) -> float:
    first_points = _points(first, name="first point cloud")
    second_points = _points(second, name="second point cloud")
    forward = _nearest_distances(first_points, second_points)
    backward = _nearest_distances(second_points, first_points)
    return float(np.sqrt(0.5 * (np.mean(forward**2) + np.mean(backward**2))))


def _comparison_rows(
    encodings: Mapping[str, _Encoding],
    bookkeeping: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
    config: ConstellationStabilityConfig,
) -> list[dict[str, Any]]:
    specs = {
        "independent_sample": ("source", "independent_sample", None),
        "translation": ("source", "translation", bookkeeping["translation"]),
        "rotation": ("source", "rotation", bookkeeping["rotation"]),
        "rotation_pca": ("pca_source", "rotation_pca", None),
    }
    rows = []
    for condition, (first_name, second_name, alignment) in specs.items():
        first = encodings[first_name]
        second = encodings[second_name]
        second_coordinates = second.coordinates
        second_reconstruction = second.quality.reconstruction[0].numpy()
        if isinstance(alignment, RigidTransform):
            second_coordinates = alignment.inverse(second_coordinates)
            second_reconstruction = alignment.inverse(second_reconstruction)
        matching = constellation_matching_metrics(
            first.coordinates,
            second_coordinates,
            coordinate_bits=int(common["coordinate_bits"]),
            radii_bins=config.repeatability_radii_bins,
        )
        rows.append(
            {
                **common,
                "condition": condition,
                "pca_pose_normalized_ablation": condition == "rotation_pca",
                "pca_frame_serialized": False,
                "known_transform_aligned": condition in {"translation", "rotation"},
                "analysis_alignment": (
                    alignment.to_json()
                    if isinstance(alignment, RigidTransform)
                    else {
                        "kind": "independent_pca_frames",
                        "first": bookkeeping["pca_source"].to_json(),
                        "second": bookkeeping["pca_rotation"].to_json(),
                    }
                    if condition == "rotation_pca"
                    else {"kind": "identity"}
                ),
                "first_stream_sha256": hashlib.sha256(first.stream).hexdigest(),
                "second_stream_sha256": hashlib.sha256(second.stream).hexdigest(),
                "first_stream_bytes": len(first.stream),
                "second_stream_bytes": len(second.stream),
                "first_encode_seconds": first.encode_seconds,
                "second_encode_seconds": second.encode_seconds,
                "first_decoder_evaluations": first.decoder_evaluations,
                "second_decoder_evaluations": second.decoder_evaluations,
                "first_selection_seed": first.selection_seed,
                "second_selection_seed": second.selection_seed,
                "first_bitstream_mode": first.bitstream_mode,
                "second_bitstream_mode": second.bitstream_mode,
                "decoded_consistency_chamfer_rmse": _chamfer_rmse(
                    first.quality.reconstruction[0].numpy(), second_reconstruction
                ),
                **matching,
            }
        )
    return rows


def _cloud_bootstrap_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value: str,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    by_cloud: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["family"]), str(row["model_id"]), int(row["sample_id"]))
        by_cloud[key].append(float(row[value]))
    if not by_cloud:
        raise ValueError("cannot summarize an empty row set")
    keys = sorted(by_cloud)
    values = np.asarray([np.mean(by_cloud[key]) for key in keys], dtype=np.float64)
    categories = np.asarray([key[0] for key in keys])
    unique_categories = np.unique(categories)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        category_draw = unique_categories[
            rng.integers(0, len(unique_categories), size=len(unique_categories))
        ]
        selected: list[int] = []
        for category in category_draw:
            candidates = np.flatnonzero(categories == category)
            selected.extend(
                candidates[
                    rng.integers(0, len(candidates), size=len(candidates))
                ].tolist()
            )
        draws[index] = float(values[np.asarray(selected)].mean())
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(draws, (alpha, 1.0 - alpha))
    return {
        "cloud_count": len(keys),
        "replicate_rows": len(rows),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "bootstrap_unit": "category_then_cloud_after_replicate_averaging",
    }


def summarize_constellation_stability(
    comparison_rows: Sequence[Mapping[str, Any]],
    cross_rows: Sequence[Mapping[str, Any]],
    config: ConstellationStabilityConfig,
) -> dict[str, Any]:
    """Aggregate Experiment 032 rows and evaluate the predeclared G-B3 gate."""

    repeatability = []
    summary_index = 0
    for split in config.splits:
        for method in config.methods:
            for condition in COMPARISON_CONDITIONS:
                selected = [
                    row
                    for row in comparison_rows
                    if row["split"] == split
                    and row["method"] == method
                    and row["condition"] == condition
                ]
                if not selected:
                    continue
                for radius in config.repeatability_radii_bins:
                    radius_rows = [
                        {
                            **row,
                            "repeatability": row["repeatability_by_radius_bins"][
                                str(radius)
                            ],
                        }
                        for row in selected
                    ]
                    interval = _cloud_bootstrap_interval(
                        radius_rows,
                        value="repeatability",
                        samples=config.bootstrap_samples,
                        confidence_level=config.confidence_level,
                        seed=config.bootstrap_seed + summary_index,
                    )
                    repeatability.append(
                        {
                            "split": split,
                            "method": method,
                            "condition": condition,
                            "radius_bins": radius,
                            **interval,
                        }
                    )
                    summary_index += 1

    hausdorff = []
    decoded_consistency = []
    for split in config.splits:
        for method in config.methods:
            for condition in COMPARISON_CONDITIONS:
                selected = [
                    row
                    for row in comparison_rows
                    if row["split"] == split
                    and row["method"] == method
                    and row["condition"] == condition
                ]
                if not selected:
                    continue
                hausdorff.append(
                    {
                        "split": split,
                        "method": method,
                        "condition": condition,
                        **_cloud_bootstrap_interval(
                            selected,
                            value="hausdorff_lattice_bins",
                            samples=config.bootstrap_samples,
                            confidence_level=config.confidence_level,
                            seed=config.bootstrap_seed + 10_000 + summary_index,
                        ),
                    }
                )
                decoded_consistency.append(
                    {
                        "split": split,
                        "method": method,
                        "condition": condition,
                        **_cloud_bootstrap_interval(
                            selected,
                            value="decoded_consistency_chamfer_rmse",
                            samples=config.bootstrap_samples,
                            confidence_level=config.confidence_level,
                            seed=config.bootstrap_seed + 15_000 + summary_index,
                        ),
                    }
                )
                summary_index += 1

    cross_decoder = []
    for split in config.splits:
        for method in config.methods:
            selected = [
                row
                for row in cross_rows
                if row["split"] == split
                and row["method"] == method
                and row["decoder_seed"] != row["cross_decoder_seed"]
            ]
            if not selected:
                continue
            cross_decoder.append(
                {
                    "split": split,
                    "method": method,
                    "metric": "fresh_rmse_degradation_percent",
                    **_cloud_bootstrap_interval(
                        selected,
                        value="fresh_rmse_degradation_percent",
                        samples=config.bootstrap_samples,
                        confidence_level=config.confidence_level,
                        seed=config.bootstrap_seed + 20_000 + summary_index,
                    ),
                }
            )
            summary_index += 1

    def repeatability_summary(condition: str) -> Mapping[str, Any] | None:
        return next(
            (
                row
                for row in repeatability
                if row["split"] == "validation"
                and row["method"] == "refiner"
                and row["condition"] == condition
                and row["radius_bins"] == config.gate_radius_bins
            ),
            None,
        )

    sample = repeatability_summary("independent_sample")
    translation = repeatability_summary("translation")
    rotation = repeatability_summary("rotation")
    rotation_pca = repeatability_summary("rotation_pca")
    cross = next(
        (
            row
            for row in cross_decoder
            if row["split"] == "validation" and row["method"] == "refiner"
        ),
        None,
    )
    checks = {
        "independent_sample_repeatability": bool(
            sample
            and sample["confidence_interval_lower"] >= config.gate_minimum_repeatability
        ),
        "translation_repeatability": bool(
            translation
            and translation["confidence_interval_lower"]
            >= config.gate_minimum_translation_repeatability
        ),
        "pca_rotation_repeatability": bool(
            rotation_pca
            and rotation_pca["confidence_interval_lower"]
            >= config.gate_minimum_repeatability
        ),
        "cross_decoder_graceful_degradation": bool(
            cross
            and cross["confidence_interval_upper"]
            <= config.gate_maximum_cross_decoder_rmse_degradation_percent
        ),
    }
    return {
        "repeatability": repeatability,
        "hausdorff_lattice_bins": hausdorff,
        "decoded_consistency_chamfer_rmse": decoded_consistency,
        "cross_decoder": cross_decoder,
        "gate_g_b3": {
            "applies_to": "validation refiner factorial",
            "radius_bins": config.gate_radius_bins,
            "thresholds": {
                "minimum_repeatability": config.gate_minimum_repeatability,
                "minimum_translation_repeatability": (
                    config.gate_minimum_translation_repeatability
                ),
                "maximum_cross_decoder_rmse_degradation_percent": (
                    config.gate_maximum_cross_decoder_rmse_degradation_percent
                ),
            },
            "raw_rotation_is_diagnostic_not_a_pass_requirement": True,
            "raw_rotation_repeatability": rotation,
            "checks": checks,
            "passes": all(checks.values()),
        },
    }


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _official_config(
    config: ConstellationStabilityConfig,
    stability: StabilityExperimentConfig,
) -> OfficialStabilityConfig:
    return OfficialStabilityConfig(
        stability_config=config.stability_config,
        stability_artifact_dir=config.stability_artifact_dir,
        position_bits=stability.coordinate_bits,
        decoder_seeds=stability.decoder_seeds,
        refiner_seeds=stability.refiner_seeds,
        methods=("fps", "refiner"),
        splits=config.splits,
        bootstrap_samples=max(100, config.bootstrap_samples),
        output_dir=config.output_dir,
    )


def run_constellation_stability(
    config: ConstellationStabilityConfig,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run the Experiment 032 stability and cross-decoder factorial."""

    device = select_device(device_name)
    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    if stability.dataset_kind != "mesh_manifest":
        raise ValueError("Experiment 032 requires independently sampled meshes")
    if not set(config.decoder_seeds) <= set(stability.decoder_seeds):
        raise ValueError("Experiment 032 decoder seeds are absent from Experiment 019")
    if not set(config.refiner_seeds) <= set(stability.refiner_seeds):
        raise ValueError("Experiment 032 refiner seeds are absent from Experiment 019")

    artifact_dir = Path(config.stability_artifact_dir)
    stability_metrics_path = artifact_dir / "stability_metrics.json"
    stability_metrics = json.loads(stability_metrics_path.read_text())
    if stability_metrics["config"] != _json_ready(asdict(stability)):
        raise RuntimeError("Experiment 019 artifact config differs from checked config")
    if not all(stability_metrics["contract_checks"].values()):
        raise RuntimeError("Experiment 019 artifact has a failed scientific contract")
    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    if data_protocol != stability_metrics["data_protocol"]:
        raise RuntimeError("Experiment 019 data identity changed before Experiment 032")

    official = _official_config(config, stability)
    decoders: dict[int, nn.Module] = {}
    decoder_hashes: dict[int, str] = {}
    refiners: dict[tuple[int, int], nn.Module] = {}
    refiner_hashes: dict[tuple[int, int], str] = {}
    model_records = []
    for decoder_seed in config.decoder_seeds:
        decoder, _, metadata = _load_models(
            stability,
            official,
            decoder_seed=decoder_seed,
            refiner_seed=None,
            device=device,
        )
        decoders[decoder_seed] = decoder
        decoder_hashes[decoder_seed] = _state_hash(decoder)
        model_records.append(
            {"decoder_seed": decoder_seed, "refiner_seed": None, **metadata}
        )
        for refiner_seed in config.refiner_seeds:
            pair_decoder, refiner, pair_metadata = _load_models(
                stability,
                official,
                decoder_seed=decoder_seed,
                refiner_seed=refiner_seed,
                device=device,
            )
            assert refiner is not None
            if _state_hash(pair_decoder) != decoder_hashes[decoder_seed]:
                raise RuntimeError("paired refiner decoder differs from sealed decoder")
            refiners[(decoder_seed, refiner_seed)] = refiner
            refiner_hashes[(decoder_seed, refiner_seed)] = _state_hash(refiner)
            model_records.append(
                {
                    "decoder_seed": decoder_seed,
                    "refiner_seed": refiner_seed,
                    **pair_metadata,
                }
            )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "032_constellation_stability",
        "config": _json_ready(asdict(config)),
        "device": device.type,
        "stability_config_sha256": file_sha256(stability_path),
        "stability_metrics_sha256": file_sha256(stability_metrics_path),
        "data_protocol": data_protocol,
        "model_records": model_records,
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing Experiment 032 run manifest differs")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    comparison_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    transform_errors: list[float] = []
    expected_bytes = expected_stream_bytes(
        stability.constellation_size, stability.coordinate_bits
    )
    evaluated_clouds = 0
    started = time.perf_counter()
    for split in config.splits:
        dataset = datasets[split]
        cloud_count = len(dataset)
        if config.max_clouds_per_split is not None:
            cloud_count = min(cloud_count, config.max_clouds_per_split)
        for cloud_index in range(cloud_count):
            evaluated_clouds += 1
            sample = dataset[cloud_index]
            metadata = _sample_metadata(sample, cloud_index)
            variants, bookkeeping = _variant_inputs(
                sample,
                split=split,
                metadata=metadata,
                config=config,
                device=device,
            )
            transform_errors.extend(
                (
                    bookkeeping["translation_inverse_max_error"],
                    bookkeeping["rotation_inverse_max_error"],
                )
            )
            for decoder_seed in config.decoder_seeds:
                decoder = decoders[decoder_seed]
                for method in config.methods:
                    method_refiner_seeds: tuple[int | None, ...] = (
                        tuple(config.refiner_seeds) if method == "refiner" else (None,)
                    )
                    for refiner_seed in method_refiner_seeds:
                        refiner = (
                            refiners[(decoder_seed, refiner_seed)]
                            if refiner_seed is not None
                            else None
                        )
                        encodings = {
                            name: _encode(
                                variant,
                                method=method,
                                decoder=decoder,
                                refiner=refiner,
                                stability=stability,
                                config=config,
                                seed_parts=(
                                    split,
                                    metadata["family"],
                                    metadata["model_id"],
                                    metadata["sample_id"],
                                    decoder_seed,
                                    refiner_seed,
                                    name,
                                ),
                                device=device,
                            )
                            for name, variant in variants.items()
                        }
                        if any(
                            len(encoding.stream) != expected_bytes
                            for encoding in encodings.values()
                        ):
                            raise RuntimeError("Experiment 032 stream size changed")
                        common = {
                            "split": split,
                            "method": method,
                            "decoder_seed": decoder_seed,
                            "refiner_seed": refiner_seed,
                            **metadata,
                            "constellation_size": stability.constellation_size,
                            "coordinate_bits": stability.coordinate_bits,
                            "representation_class": encodings[
                                "source"
                            ].representation_class,
                            "actual_stream_bytes": expected_bytes,
                            "actual_stream_bpp": (
                                8.0 * expected_bytes / stability.num_points
                            ),
                            "source_only_encoding": True,
                        }
                        comparison_rows.extend(
                            _comparison_rows(
                                encodings,
                                bookkeeping,
                                common=common,
                                config=config,
                            )
                        )

                        source_encoding = encodings["source"]
                        matched_fresh_rmse = math.sqrt(
                            max(source_encoding.quality.fresh_chamfer_mse, 0.0)
                        )
                        matched_source_rmse = math.sqrt(
                            max(source_encoding.quality.source_chamfer_mse, 0.0)
                        )
                        for cross_decoder_seed, cross_decoder in decoders.items():
                            quality = evaluate_cross_decoder(
                                cross_decoder,
                                source_encoding.stream,
                                source_points=variants["source"].source,
                                fresh_points=variants["source"].fresh,
                                device=device,
                                distance_chunk_size=stability.distance_chunk_size,
                            )
                            cross_source_rmse = math.sqrt(
                                max(quality.source_chamfer_mse, 0.0)
                            )
                            cross_fresh_rmse = math.sqrt(
                                max(quality.fresh_chamfer_mse, 0.0)
                            )
                            cross_rows.append(
                                {
                                    **common,
                                    "cross_decoder_seed": cross_decoder_seed,
                                    "stream_sha256": hashlib.sha256(
                                        source_encoding.stream
                                    ).hexdigest(),
                                    "stream_bytes": len(source_encoding.stream),
                                    "coordinate_only_transfer": True,
                                    "bitstream_mode": (source_encoding.bitstream_mode),
                                    "selection_seed": (source_encoding.selection_seed),
                                    "decoder_evaluations": (
                                        source_encoding.decoder_evaluations
                                    ),
                                    "encode_seconds": (source_encoding.encode_seconds),
                                    "matched_source_chamfer_mse": (
                                        source_encoding.quality.source_chamfer_mse
                                    ),
                                    "matched_fresh_chamfer_mse": (
                                        source_encoding.quality.fresh_chamfer_mse
                                    ),
                                    "cross_source_chamfer_mse": (
                                        quality.source_chamfer_mse
                                    ),
                                    "cross_fresh_chamfer_mse": (
                                        quality.fresh_chamfer_mse
                                    ),
                                    "source_rmse_degradation_percent": 100.0
                                    * (cross_source_rmse - matched_source_rmse)
                                    / max(matched_source_rmse, 1e-12),
                                    "fresh_rmse_degradation_percent": 100.0
                                    * (cross_fresh_rmse - matched_fresh_rmse)
                                    / max(matched_fresh_rmse, 1e-12),
                                    "decoded_cross_decoder_consistency_chamfer_rmse": (
                                        _chamfer_rmse(
                                            source_encoding.quality.reconstruction[
                                                0
                                            ].numpy(),
                                            quality.reconstruction[0].numpy(),
                                        )
                                    ),
                                }
                            )

    for decoder_seed, decoder in decoders.items():
        if _state_hash(decoder) != decoder_hashes[decoder_seed]:
            raise RuntimeError("frozen decoder changed during Experiment 032")
    for key, refiner in refiners.items():
        if _state_hash(refiner) != refiner_hashes[key]:
            raise RuntimeError("frozen refiner changed during Experiment 032")

    statistics = summarize_constellation_stability(comparison_rows, cross_rows, config)
    comparison_path = output_dir / "constellation_pairs.jsonl"
    cross_path = output_dir / "cross_decoder.jsonl"
    _write_jsonl(comparison_path, comparison_rows)
    _write_jsonl(cross_path, cross_rows)
    encoder_cells_per_decoder = len(config.methods) - 1 + len(config.refiner_seeds)
    expected_comparison_rows = (
        evaluated_clouds
        * len(config.decoder_seeds)
        * encoder_cells_per_decoder
        * len(COMPARISON_CONDITIONS)
    )
    expected_cross_rows = (
        evaluated_clouds
        * len(config.decoder_seeds)
        * encoder_cells_per_decoder
        * len(config.decoder_seeds)
    )
    contract_checks = {
        "experiment_019_contract_passed": True,
        "data_protocol_unchanged": True,
        "two_independent_samples_per_mesh": True,
        "known_transform_round_trip": bool(
            transform_errors and max(transform_errors) <= 1e-6
        ),
        "all_messages_are_coordinate_only": bool(
            cross_rows and all(row["coordinate_only_transfer"] for row in cross_rows)
        ),
        "actual_stream_bytes_present": bool(
            comparison_rows
            and all(
                row["actual_stream_bytes"] == expected_bytes for row in comparison_rows
            )
        ),
        "exact_stream_round_trip_and_lattice": True,
        "pca_frames_not_serialized": bool(
            comparison_rows
            and all(not row["pca_frame_serialized"] for row in comparison_rows)
        ),
        "decoder_hashes_unchanged": True,
        "refiner_hashes_unchanged": True,
        "complete_factorial": bool(
            len(comparison_rows) == expected_comparison_rows
            and len(cross_rows) == expected_cross_rows
        ),
        "source_only_encoding": bool(
            comparison_rows
            and all(row["source_only_encoding"] for row in comparison_rows)
        ),
    }
    result = {
        "experiment": "032_constellation_stability",
        "config": _json_ready(asdict(config)),
        "comparison_rows": len(comparison_rows),
        "expected_comparison_rows": expected_comparison_rows,
        "cross_decoder_rows": len(cross_rows),
        "expected_cross_decoder_rows": expected_cross_rows,
        "expected_stream_bytes": expected_bytes,
        "contract_checks": contract_checks,
        "statistics": statistics,
        "elapsed_seconds": time.perf_counter() - started,
        "comparison_path": str(comparison_path),
        "cross_decoder_path": str(cross_path),
    }
    if not all(contract_checks.values()):
        raise RuntimeError("Experiment 032 scientific contract failed")
    metrics_path = output_dir / "constellation_stability_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_032_constellation_stability_smoke.json"),
    )
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--stability-artifact-dir", type=Path)
    parser.add_argument("--max-clouds-per-split", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = ConstellationStabilityConfig.from_json(args.config)
    if args.stability_artifact_dir is not None:
        config = replace(
            config, stability_artifact_dir=str(args.stability_artifact_dir)
        )
    if args.max_clouds_per_split is not None:
        config = replace(config, max_clouds_per_split=args.max_clouds_per_split)
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    result = run_constellation_stability(config, device_name=args.device)
    print(
        json.dumps(
            {
                "comparison_rows": result["comparison_rows"],
                "cross_decoder_rows": result["cross_decoder_rows"],
                "gate_g_b3_passes": result["statistics"]["gate_g_b3"]["passes"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(
                    Path(config.output_dir) / "constellation_stability_metrics.json"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
