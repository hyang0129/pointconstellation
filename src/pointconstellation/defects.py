"""Deterministic local defect injection for point-cloud anomaly benchmarks.

The functions in this module operate on coordinates and optional source-visible
normals only.  They do not inspect decoder outputs, target resamples, category
labels, or point order when choosing a defect.  A stable cloud identifier is
folded into the random seed so that defects are reproducible independently of
dataset traversal order.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFECT_TYPES = ("dent", "bump", "hole", "thin_spur", "surface_noise")
DefectType = Literal["dent", "bump", "hole", "thin_spur", "surface_noise"]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
LabelArray = NDArray[np.uint8]
CODEC_DOMAIN_LOWER = -1.0
CODEC_DOMAIN_UPPER = 1.0


def _require_codec_domain(points: FloatArray, *, context: str) -> None:
    minimum = float(points.min())
    maximum = float(points.max())
    if minimum < CODEC_DOMAIN_LOWER or maximum > CODEC_DOMAIN_UPPER:
        raise ValueError(
            f"{context} must lie in the declared codec domain [-1, 1]; "
            f"observed coordinate range [{minimum:.9g}, {maximum:.9g}]"
        )


@dataclass(frozen=True)
class DefectInjectionConfig:
    """Geometry and size bounds shared by all deterministic defect types."""

    minimum_fraction: float = 0.01
    maximum_fraction: float = 0.05
    displacement_scale: float = 2.5
    spur_length_scale: float = 6.0
    spur_radius_scale: float = 0.12
    noise_scale: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_fraction <= self.maximum_fraction < 0.5:
            raise ValueError(
                "defect fractions must satisfy 0 < minimum <= maximum < 0.5"
            )
        positive = (
            self.displacement_scale,
            self.spur_length_scale,
            self.spur_radius_scale,
            self.noise_scale,
        )
        if min(positive) <= 0.0 or not all(math.isfinite(value) for value in positive):
            raise ValueError("defect geometry scales must be finite and positive")

    @classmethod
    def from_json(cls, path: Path) -> DefectInjectionConfig:
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class DefectResult:
    """One injected cloud and labels aligned with its returned point array.

    Every defect preserves the input cardinality.  For a hole, absent samples
    cannot carry labels, so the removed samples are replaced by deterministic
    resamples from outside the patch and the label is assigned to the equally
    sized nearest surviving rim.  ``removed_count`` records the actual number
    of deleted samples.  This makes the point task well-defined without
    pretending that a finite sampled hole contains observed defective points.
    """

    points: FloatArray
    point_labels: LabelArray
    cloud_label: int
    defect_type: str
    requested_fraction: float
    declared_fraction: float
    defective_count: int
    original_point_count: int
    removed_count: int
    source_indices: IntArray
    seed: int
    domain_scale_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1:] != (3,):
            raise ValueError("defect points must have shape (N, 3)")
        if not np.isfinite(self.points).all():
            raise ValueError("defect points must be finite")
        _require_codec_domain(self.points, context="injected defect points")
        if self.point_labels.shape != (len(self.points),):
            raise ValueError("point labels must align with defect points")
        if self.source_indices.shape != (len(self.points),):
            raise ValueError("source indices must align with defect points")
        if np.any((self.point_labels != 0) & (self.point_labels != 1)):
            raise ValueError("point labels must be binary")
        if int(self.point_labels.sum()) != self.defective_count:
            raise ValueError("defective_count differs from point labels")
        if self.cloud_label not in {0, 1}:
            raise ValueError("cloud label must be binary")
        if self.cloud_label != int(self.defect_type != "none"):
            raise ValueError("cloud label differs from defect type")
        if self.removed_count < 0:
            raise ValueError("removed_count cannot be negative")
        if not math.isfinite(self.domain_scale_factor) or not (
            0.0 < self.domain_scale_factor <= 1.0
        ):
            raise ValueError("domain_scale_factor must lie in (0, 1]")
        if self.defect_type in {"none", "hole"} and self.domain_scale_factor != 1.0:
            raise ValueError(
                "non-displacement defects must have domain_scale_factor equal to 1"
            )
        if len(self.points) != self.original_point_count:
            raise ValueError("defect injection must preserve source cardinality")
        if self.defect_type == "hole" and self.removed_count != self.defective_count:
            raise ValueError("hole removed_count differs from defective_count")
        if self.defect_type != "hole" and self.removed_count:
            raise ValueError("only a hole may remove source samples")


def defect_seed(base_seed: int, cloud_id: str, defect_type: str) -> int:
    """Derive a traversal-order-independent 64-bit seed for one cloud/defect."""

    if base_seed < 0:
        raise ValueError("base seed must be nonnegative")
    if not cloud_id:
        raise ValueError("cloud_id must be nonempty")
    digest = hashlib.sha256(f"{base_seed}:{cloud_id}:{defect_type}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _points(points: ArrayLike) -> FloatArray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1:] != (3,) or len(array) < 8:
        raise ValueError("points must have shape (N, 3) with N >= 8")
    if not np.isfinite(array).all():
        raise ValueError("points must be finite")
    _require_codec_domain(array, context="defect source points")
    return array


def _unit_normals(points: FloatArray, normals: ArrayLike | None) -> FloatArray:
    if normals is None:
        centered = points.astype(np.float64) - points.mean(axis=0, dtype=np.float64)
        lengths = np.linalg.norm(centered, axis=1)
        fallback = np.zeros_like(centered)
        fallback[:, 2] = 1.0
        output = np.divide(
            centered,
            lengths[:, None],
            out=fallback,
            where=lengths[:, None] > 1e-12,
        )
        return output.astype(np.float32)
    array = np.asarray(normals, dtype=np.float64)
    if array.shape != points.shape or not np.isfinite(array).all():
        raise ValueError("normals must be finite and have the same shape as points")
    lengths = np.linalg.norm(array, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("normals must have nonzero length")
    return (array / lengths[:, None]).astype(np.float32)


def _defective_count(
    count: int,
    fraction: float,
    *,
    minimum_fraction: float,
    maximum_fraction: float,
) -> int:
    minimum = max(1, int(math.ceil(count * minimum_fraction)))
    maximum = int(math.floor(count * maximum_fraction))
    if maximum < minimum:
        raise ValueError(
            "point cloud is too small to realize the configured fraction interval"
        )
    return min(maximum, max(minimum, int(round(count * fraction))))


def _local_patch(points: FloatArray, count: int, rng: np.random.Generator) -> IntArray:
    canonical = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    anchor = int(canonical[int(rng.integers(0, len(points)))])
    distances = np.sum(
        (points.astype(np.float64) - points[anchor].astype(np.float64)) ** 2,
        axis=1,
    )
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], distances))
    return order[:count].astype(np.int64)


def _local_scale(points: FloatArray, patch: IntArray) -> float:
    if len(points) < 2:
        return 1.0
    anchor = points[int(patch[0])].astype(np.float64)
    patch_distances = np.linalg.norm(points[patch].astype(np.float64) - anchor, axis=1)
    positive = patch_distances[patch_distances > 1e-12]
    if len(positive):
        scale = float(np.quantile(positive, 0.75))
    else:
        extent = np.linalg.norm(
            points.astype(np.float64) - points.mean(axis=0, dtype=np.float64), axis=1
        )
        scale = float(np.quantile(extent, 0.5))
    return max(scale, np.finfo(np.float32).eps)


def _patch_weights(points: FloatArray, patch: IntArray) -> FloatArray:
    anchor = points[int(patch[0])].astype(np.float64)
    distances = np.linalg.norm(points[patch].astype(np.float64) - anchor, axis=1)
    radius = max(float(distances.max()), np.finfo(np.float32).eps)
    # The nonzero shoulder creates a coherent interior displacement while the
    # cosine taper avoids an artificial step at the selected patch boundary.
    weights = 0.35 + 0.65 * 0.5 * (1.0 + np.cos(np.pi * distances / radius))
    return weights.astype(np.float32)


def _domain_preserving_displacement(
    base: FloatArray, displacement: NDArray[np.float64]
) -> tuple[FloatArray, float]:
    """Uniformly attenuate one generated displacement to stay in the codec cube.

    A single scale preserves the generated defect's directions and relative
    taper/radius.  This is a bound on the injection magnitude, not coordinate
    clipping: coordinates that already fit retain scale one, and an attenuated
    injection records its realized multiplier in ``DefectResult``.
    """

    original = np.asarray(base, dtype=np.float64)
    delta = np.asarray(displacement, dtype=np.float64)
    if (
        original.shape != delta.shape
        or original.ndim != 2
        or original.shape[1:] != (3,)
    ):
        raise ValueError("defect displacement must align with base points")
    if not np.isfinite(delta).all():
        raise ValueError("defect displacement must be finite")
    if not np.any(delta):
        raise ValueError("defect displacement must be nonzero")

    limits = [1.0]
    positive = delta > 0.0
    if positive.any():
        limits.append(
            float(np.min((CODEC_DOMAIN_UPPER - original)[positive] / delta[positive]))
        )
    negative = delta < 0.0
    if negative.any():
        limits.append(
            float(
                np.min(
                    (original - CODEC_DOMAIN_LOWER)[negative] / -delta[negative]
                )
            )
        )
    scale = min(limits)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "cannot realize a nonzero defect displacement inside the declared "
            "codec domain [-1, 1]"
        )
    scale = min(1.0, scale)
    if scale < 1.0:
        scale = float(np.nextafter(scale, 0.0))
    output = (original + scale * delta).astype(np.float32)
    _require_codec_domain(output, context="domain-preserving defect points")
    return output, scale


def _control(points: FloatArray, *, seed: int) -> DefectResult:
    count = len(points)
    return DefectResult(
        points=points.copy(),
        point_labels=np.zeros(count, dtype=np.uint8),
        cloud_label=0,
        defect_type="none",
        requested_fraction=0.0,
        declared_fraction=0.0,
        defective_count=0,
        original_point_count=count,
        removed_count=0,
        source_indices=np.arange(count, dtype=np.int64),
        seed=seed,
        domain_scale_factor=1.0,
    )


def inject_defect(
    points: ArrayLike,
    defect_type: DefectType | Literal["none"] | None,
    *,
    seed: int,
    fraction: float | None = None,
    normals: ArrayLike | None = None,
    config: DefectInjectionConfig | None = None,
) -> DefectResult:
    """Inject one deterministic local defect into a finite point sample.

    ``fraction=None`` samples a continuous fraction uniformly inside the
    configured 1--5% interval.  Supplying a fraction fixes the requested size;
    the exact integer count and its fraction of the original sample are exposed
    as ``defective_count`` and ``declared_fraction``.  Every returned condition
    has exactly the source cardinality: a thin spur relocates its selected patch
    and a hole deterministically resamples surviving source points to replace
    the removed samples.
    """

    source = _points(points)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    defect_name = "none" if defect_type is None else str(defect_type)
    if defect_name == "none":
        if fraction not in {None, 0}:
            raise ValueError("an undefected control cannot have a defect fraction")
        return _control(source, seed=seed)
    if defect_name not in DEFECT_TYPES:
        raise ValueError(f"defect_type must be one of {DEFECT_TYPES} or 'none'")

    settings = config or DefectInjectionConfig()
    rng = np.random.default_rng(seed)
    requested = (
        float(rng.uniform(settings.minimum_fraction, settings.maximum_fraction))
        if fraction is None
        else float(fraction)
    )
    if not settings.minimum_fraction <= requested <= settings.maximum_fraction:
        raise ValueError("fraction lies outside the configured defect-size interval")
    defective_count = _defective_count(
        len(source),
        requested,
        minimum_fraction=settings.minimum_fraction,
        maximum_fraction=settings.maximum_fraction,
    )
    patch = _local_patch(source, defective_count, rng)
    local_scale = _local_scale(source, patch)
    unit_normals = _unit_normals(source, normals)
    labels = np.zeros(len(source), dtype=np.uint8)
    source_indices = np.arange(len(source), dtype=np.int64)
    output = source.copy()
    removed_count = 0
    domain_scale_factor = 1.0

    if defect_name in {"dent", "bump"}:
        direction = -1.0 if defect_name == "dent" else 1.0
        displacement = (
            direction
            * settings.displacement_scale
            * local_scale
            * _patch_weights(source, patch)
        )
        output[patch], domain_scale_factor = _domain_preserving_displacement(
            source[patch],
            displacement[:, None].astype(np.float64)
            * unit_normals[patch].astype(np.float64),
        )
        labels[patch] = 1
    elif defect_name == "surface_noise":
        noise = rng.normal(size=(defective_count, 3))
        noise /= np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-12)
        amplitudes = rng.uniform(0.5, 1.0, size=(defective_count, 1))
        output[patch], domain_scale_factor = _domain_preserving_displacement(
            source[patch], settings.noise_scale * local_scale * amplitudes * noise
        )
        labels[patch] = 1
    elif defect_name == "thin_spur":
        anchor = int(patch[0])
        axis = unit_normals[anchor].astype(np.float64)
        fallback = np.asarray([1.0, 0.0, 0.0])
        tangent = np.cross(axis, fallback)
        if np.linalg.norm(tangent) <= 1e-8:
            tangent = np.cross(axis, np.asarray([0.0, 1.0, 0.0]))
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(axis, tangent)
        distances = np.linspace(0.3, 1.0, defective_count)[:, None]
        angles = rng.uniform(0.0, 2.0 * np.pi, size=(defective_count, 1))
        radii = (
            settings.spur_radius_scale * local_scale * rng.random((defective_count, 1))
        )
        spur_displacement = (
            settings.spur_length_scale * local_scale * distances * axis[None]
            + radii
            * (np.cos(angles) * tangent[None] + np.sin(angles) * bitangent[None])
        )
        spur_target = source[anchor].astype(np.float64)[None] + spur_displacement
        output[patch], domain_scale_factor = _domain_preserving_displacement(
            source[patch], spur_target - source[patch].astype(np.float64)
        )
        labels[patch] = 1
    else:
        removed_count = defective_count
        keep = np.ones(len(source), dtype=bool)
        keep[patch] = False
        survivor_indices = np.flatnonzero(keep).astype(np.int64)
        survivors = source[survivor_indices]
        canonical = np.lexsort(
            (survivors[:, 2], survivors[:, 1], survivors[:, 0])
        )
        replacement_positions = rng.choice(
            len(canonical), size=defective_count, replace=False
        )
        replacement_indices = survivor_indices[canonical[replacement_positions]]
        output = np.concatenate((survivors, source[replacement_indices]), axis=0)
        source_indices = np.concatenate((survivor_indices, replacement_indices))
        anchor = source[int(patch[0])].astype(np.float64)
        rim_distances = np.linalg.norm(survivors.astype(np.float64) - anchor, axis=1)
        rim = np.lexsort(
            (
                survivors[:, 2],
                survivors[:, 1],
                survivors[:, 0],
                rim_distances,
            )
        )[:defective_count]
        labels = np.zeros(len(output), dtype=np.uint8)
        labels[rim] = 1

    output = output.astype(np.float32, copy=False)
    _require_codec_domain(output, context=f"injected {defect_name} points")
    return DefectResult(
        points=output,
        point_labels=labels,
        cloud_label=1,
        defect_type=defect_name,
        requested_fraction=requested,
        declared_fraction=defective_count / len(source),
        defective_count=defective_count,
        original_point_count=len(source),
        removed_count=removed_count,
        source_indices=source_indices,
        seed=seed,
        domain_scale_factor=domain_scale_factor,
    )


def inject_defect_for_cloud(
    points: ArrayLike,
    defect_type: DefectType | Literal["none"] | None,
    *,
    base_seed: int,
    cloud_id: str,
    fraction: float | None = None,
    normals: ArrayLike | None = None,
    config: DefectInjectionConfig | None = None,
) -> DefectResult:
    """Inject a defect using a stable seed derived from the cloud identity."""

    name = "none" if defect_type is None else str(defect_type)
    return inject_defect(
        points,
        defect_type,
        seed=defect_seed(base_seed, cloud_id, name),
        fraction=fraction,
        normals=normals,
        config=config,
    )


def transfer_point_labels(
    reference_points: ArrayLike,
    reference_labels: ArrayLike,
    query_points: ArrayLike,
    *,
    chunk_size: int = 512,
) -> LabelArray:
    """Transfer binary labels from the nearest finite reference sample.

    This is explicitly a nearest-sampled-target proxy.  It is not a label on an
    unobserved continuous surface and is recorded as such by Experiment 041.
    """

    reference = np.asarray(reference_points, dtype=np.float64)
    query = np.asarray(query_points, dtype=np.float64)
    for name, value in (("reference", reference), ("query", query)):
        if value.ndim != 2 or value.shape[1:] != (3,) or not len(value):
            raise ValueError(f"{name} points must have shape (N, 3) with N > 0")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} points must be finite")
    labels = np.asarray(reference_labels, dtype=np.uint8)
    if labels.shape != (len(reference),) or np.any((labels != 0) & (labels != 1)):
        raise ValueError("reference labels must be binary and align with points")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    transferred = np.empty(len(query), dtype=np.uint8)
    reference_norms = np.einsum("ij,ij->i", reference, reference)
    canonical = np.lexsort((reference[:, 2], reference[:, 1], reference[:, 0]))
    canonical_ranks = np.empty(len(reference), dtype=np.int64)
    canonical_ranks[canonical] = np.arange(len(reference))
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        values = query[start:stop]
        squared = (
            np.einsum("ij,ij->i", values, values)[:, None]
            + reference_norms[None]
            - 2.0 * values @ reference.T
        )
        # argmin is stable for the canonical reference order; lexsort makes the
        # tie behavior explicit for noncanonical caller inputs.
        for offset, row in enumerate(squared):
            nearest = np.lexsort((canonical_ranks, row))[0]
            transferred[start + offset] = labels[nearest]
    return transferred


def size_stratum(fraction: float) -> str:
    """Map a declared 1--5% fraction to a predeclared reporting stratum."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if fraction < 0.02:
        return "small_1_2pct"
    if fraction < 0.04:
        return "medium_2_4pct"
    return "large_4_5pct"


__all__ = [
    "CODEC_DOMAIN_LOWER",
    "CODEC_DOMAIN_UPPER",
    "DEFECT_TYPES",
    "DefectInjectionConfig",
    "DefectResult",
    "DefectType",
    "defect_seed",
    "inject_defect",
    "inject_defect_for_cloud",
    "size_stratum",
    "transfer_point_labels",
]
