"""Independent samples and analytic geometry for fixed procedural surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.data.procedural import FAMILIES

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]

_SPLIT_CODES = {"train": 0, "validation": 1, "id_test": 2, "parameter_ood": 3}
_ROLE_CODES = {"x_a": 11, "x_b": 13, "x_c": 17}


@dataclass(frozen=True)
class ProceduralSurface:
    """One centered surface before and after a fixed rigid normalization."""

    family: str
    parameters: tuple[float, ...]
    rotation: FloatArray
    scale: float
    sample_id: int
    split: str

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown procedural family: {self.family}")
        rotation = np.asarray(self.rotation)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("surface rotation must be a finite 3 x 3 matrix")
        if self.scale <= 0 or not np.isfinite(self.scale):
            raise ValueError("surface scale must be finite and positive")


@dataclass(frozen=True)
class SurfacePointSample:
    points: FloatArray
    normals: FloatArray
    boundary_mask: BoolArray
    thin_structure_mask: BoolArray


@dataclass(frozen=True)
class ProceduralSurfaceTriplet:
    surface: ProceduralSurface
    x_a: SurfacePointSample
    x_b: SurfacePointSample
    x_c: SurfacePointSample


def _rotation(rng: np.random.Generator) -> NDArray[np.float64]:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1.0
    return matrix


def generate_procedural_surface(
    sample_id: int,
    *,
    seed: int = 7,
    split: str = "train",
) -> ProceduralSurface:
    """Generate fixed parameters independently of any surface point draw."""

    if split not in _SPLIT_CODES:
        raise ValueError(f"unknown split: {split}")
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, _SPLIT_CODES[split], sample_id, 31))
    )
    family = FAMILIES[sample_id % len(FAMILIES)]
    ood = split == "parameter_ood"
    if family == "plane":
        width = rng.uniform(1.5, 2.0) if ood else rng.uniform(0.8, 1.5)
        height = rng.uniform(0.2, 0.5) if ood else rng.uniform(0.6, 1.3)
        parameters = (float(width), float(height))
        scale = float(np.hypot(width, height))
    elif family == "corner":
        reach = rng.uniform(1.3, 1.8) if ood else rng.uniform(0.7, 1.2)
        parameters = (float(reach),)
        scale = float(np.sqrt(2.0) * reach)
    elif family == "box":
        low, high = (0.2, 1.6) if ood else (0.55, 1.1)
        extents = rng.uniform(low, high, size=3)
        parameters = tuple(float(value) for value in extents)
        scale = float(np.linalg.norm(extents))
    elif family == "sphere":
        radius = rng.uniform(1.2, 1.6) if ood else rng.uniform(0.7, 1.1)
        parameters = (float(radius),)
        scale = float(radius)
    elif family == "cylinder":
        radius = rng.uniform(0.25, 0.4) if ood else rng.uniform(0.55, 0.9)
        half_height = rng.uniform(1.2, 1.8) if ood else rng.uniform(0.6, 1.1)
        parameters = (float(radius), float(half_height))
        scale = float(np.hypot(radius, half_height))
    elif family == "beam":
        thickness = rng.uniform(0.025, 0.08) if ood else rng.uniform(0.08, 0.2)
        extents = np.asarray([rng.uniform(0.8, 1.4), thickness, rng.uniform(0.4, 0.8)])
        parameters = tuple(float(value) for value in extents)
        scale = float(np.linalg.norm(extents))
    else:
        separation = rng.uniform(1.1, 1.5) if ood else rng.uniform(0.65, 1.0)
        radius = rng.uniform(0.2, 0.35)
        parameters = (float(separation), float(radius))
        scale = float(separation + radius)
    return ProceduralSurface(
        family=family,
        parameters=parameters,
        rotation=_rotation(rng).astype(np.float32),
        scale=scale,
        sample_id=sample_id,
        split=split,
    )


def _unit_vectors(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    vectors = rng.normal(size=(count, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)


def _sample_box(
    rng: np.random.Generator,
    count: int,
    extents: NDArray[np.float64],
    *,
    boundary_band: float,
    thin: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], BoolArray, BoolArray]:
    faces = rng.integers(0, 6, size=count)
    points = rng.uniform(-1.0, 1.0, size=(count, 3)) * extents
    normals = np.zeros((count, 3), dtype=np.float64)
    axes = faces // 2
    signs = np.where(faces % 2, 1.0, -1.0)
    rows = np.arange(count)
    points[rows, axes] = signs * extents[axes]
    normals[rows, axes] = signs
    normalized_margin = (extents[None, :] - np.abs(points)) / extents[None, :]
    on_face = np.zeros((count, 3), dtype=bool)
    on_face[rows, axes] = True
    boundary = np.any((normalized_margin <= boundary_band) & ~on_face, axis=1)
    return points, normals, boundary, np.full(count, thin, dtype=bool)


def sample_procedural_surface(
    surface: ProceduralSurface,
    count: int,
    *,
    seed: int,
    role: str,
    boundary_band: float = 0.1,
) -> SurfacePointSample:
    """Draw one deterministic role-specific point set from a fixed surface."""

    if count < 8:
        raise ValueError("sample count must be at least 8")
    if role not in _ROLE_CODES:
        raise ValueError(f"unknown sample role: {role}")
    if not 0.0 < boundary_band < 0.5:
        raise ValueError("boundary_band must be in (0, 0.5)")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            (
                seed,
                _SPLIT_CODES[surface.split],
                surface.sample_id,
                _ROLE_CODES[role],
            )
        )
    )
    family = surface.family
    parameters = surface.parameters
    boundary = np.zeros(count, dtype=bool)
    thin = np.zeros(count, dtype=bool)
    if family == "plane":
        width, height = parameters
        uv = rng.uniform((-width, -height), (width, height), size=(count, 2))
        points = np.column_stack((uv, np.zeros(count)))
        normals = np.tile((0.0, 0.0, 1.0), (count, 1))
        boundary = (np.abs(uv[:, 0]) >= width * (1.0 - boundary_band)) | (
            np.abs(uv[:, 1]) >= height * (1.0 - boundary_band)
        )
    elif family == "corner":
        (reach,) = parameters
        first = count // 2
        second = count - first
        xy = rng.uniform(-reach, reach, size=(first, 2))
        xz = rng.uniform(-reach, reach, size=(second, 2))
        points = np.vstack(
            (
                np.column_stack((xy, np.zeros(first))),
                np.column_stack((xz[:, 0], np.zeros(second), xz[:, 1])),
            )
        )
        normals = np.vstack(
            (
                np.tile((0.0, 0.0, 1.0), (first, 1)),
                np.tile((0.0, 1.0, 0.0), (second, 1)),
            )
        )
        boundary[:first] = (
            (np.abs(xy[:, 0]) >= reach * (1.0 - boundary_band))
            | (np.abs(xy[:, 1]) >= reach * (1.0 - boundary_band))
            | (np.abs(xy[:, 1]) <= reach * boundary_band)
        )
        boundary[first:] = (
            (np.abs(xz[:, 0]) >= reach * (1.0 - boundary_band))
            | (np.abs(xz[:, 1]) >= reach * (1.0 - boundary_band))
            | (np.abs(xz[:, 1]) <= reach * boundary_band)
        )
    elif family in {"box", "beam"}:
        points, normals, boundary, thin = _sample_box(
            rng,
            count,
            np.asarray(parameters),
            boundary_band=boundary_band,
            thin=family == "beam",
        )
    elif family == "sphere":
        (radius,) = parameters
        normals = _unit_vectors(rng, count)
        points = radius * normals
    elif family == "cylinder":
        radius, half_height = parameters
        cap = rng.random(count) < 0.2
        theta = rng.uniform(0.0, 2.0 * np.pi, count)
        points = np.empty((count, 3), dtype=np.float64)
        normals = np.zeros((count, 3), dtype=np.float64)
        side = ~cap
        points[side, 0] = radius * np.cos(theta[side])
        points[side, 1] = radius * np.sin(theta[side])
        points[side, 2] = rng.uniform(-half_height, half_height, side.sum())
        normals[side, 0] = np.cos(theta[side])
        normals[side, 1] = np.sin(theta[side])
        cap_radius = radius * np.sqrt(rng.random(cap.sum()))
        points[cap, 0] = cap_radius * np.cos(theta[cap])
        points[cap, 1] = cap_radius * np.sin(theta[cap])
        signs = rng.choice((-1.0, 1.0), cap.sum())
        points[cap, 2] = signs * half_height
        normals[cap, 2] = signs
        boundary[side] = np.abs(points[side, 2]) >= half_height * (1.0 - boundary_band)
        boundary[cap] = cap_radius >= radius * (1.0 - boundary_band)
    else:
        separation, radius = parameters
        first = count // 2
        normals_a = _unit_vectors(rng, first)
        normals_b = _unit_vectors(rng, count - first)
        points = np.vstack(
            (
                radius * normals_a + np.array([-separation, 0.0, 0.0]),
                radius * normals_b + np.array([separation, 0.0, 0.0]),
            )
        )
        normals = np.vstack((normals_a, normals_b))
        thin[:] = True

    rotation = np.asarray(surface.rotation, dtype=np.float64)
    points = points @ rotation.T / surface.scale
    normals = normals @ rotation.T
    permutation = rng.permutation(count)
    return SurfacePointSample(
        points[permutation].astype(np.float32),
        normals[permutation].astype(np.float32),
        boundary[permutation],
        thin[permutation],
    )


def generate_surface_triplet(
    sample_id: int,
    *,
    num_points: int = 1024,
    seed: int = 7,
    split: str = "train",
    boundary_band: float = 0.1,
) -> ProceduralSurfaceTriplet:
    """Generate independent ``X_a``, ``X_b``, and ``X_c`` samples."""

    surface = generate_procedural_surface(sample_id, seed=seed, split=split)
    samples = {
        role: sample_procedural_surface(
            surface,
            num_points,
            seed=seed,
            role=role,
            boundary_band=boundary_band,
        )
        for role in _ROLE_CODES
    }
    return ProceduralSurfaceTriplet(
        surface=surface,
        x_a=samples["x_a"],
        x_b=samples["x_b"],
        x_c=samples["x_c"],
    )


def _local_points(points: ArrayLike, surface: ProceduralSurface) -> NDArray[np.float64]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("points must have shape (N, 3) with N > 0")
    if not np.isfinite(values).all():
        raise ValueError("points must contain only finite values")
    return values @ np.asarray(surface.rotation, dtype=np.float64) * surface.scale


def analytic_surface_distances(
    points: ArrayLike, surface: ProceduralSurface
) -> NDArray[np.float64]:
    """Return exact distances to the fixed finite procedural surface."""

    local = _local_points(points, surface)
    family = surface.family
    parameters = surface.parameters
    if family == "plane":
        width, height = parameters
        outside = np.column_stack(
            (
                np.maximum(np.abs(local[:, 0]) - width, 0.0),
                np.maximum(np.abs(local[:, 1]) - height, 0.0),
                local[:, 2],
            )
        )
        distances = np.linalg.norm(outside, axis=1)
    elif family == "corner":
        (reach,) = parameters

        def patch_distance(
            first: int, second: int, orthogonal: int
        ) -> NDArray[np.float64]:
            outside = np.column_stack(
                (
                    np.maximum(np.abs(local[:, first]) - reach, 0.0),
                    np.maximum(np.abs(local[:, second]) - reach, 0.0),
                    local[:, orthogonal],
                )
            )
            return np.linalg.norm(outside, axis=1)

        distances = np.minimum(patch_distance(0, 1, 2), patch_distance(0, 2, 1))
    elif family in {"box", "beam"}:
        extents = np.asarray(parameters)
        offset = np.abs(local) - extents[None, :]
        outside = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
        inside = -np.max(offset, axis=1)
        distances = np.where(np.any(offset > 0.0, axis=1), outside, inside)
    elif family == "sphere":
        (radius,) = parameters
        distances = np.abs(np.linalg.norm(local, axis=1) - radius)
    elif family == "cylinder":
        radius, half_height = parameters
        radial = np.linalg.norm(local[:, :2], axis=1)
        offset = np.column_stack((radial - radius, np.abs(local[:, 2]) - half_height))
        signed = np.linalg.norm(np.maximum(offset, 0.0), axis=1) + np.minimum(
            np.maximum(offset[:, 0], offset[:, 1]), 0.0
        )
        distances = np.abs(signed)
    else:
        separation, radius = parameters
        first = np.linalg.norm(local - np.array([-separation, 0.0, 0.0]), axis=1)
        second = np.linalg.norm(local - np.array([separation, 0.0, 0.0]), axis=1)
        distances = np.minimum(np.abs(first - radius), np.abs(second - radius))
    return distances / surface.scale


class ProceduralSurfaceDataset:
    """PyTorch-compatible fixed-surface dataset with three independent samples."""

    def __init__(
        self,
        size: int,
        *,
        num_points: int = 1024,
        seed: int = 7,
        split: str = "train",
        training_target: str = "exact_sample",
        boundary_band: float = 0.1,
    ) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        if training_target not in {"exact_sample", "independent_resampling"}:
            raise ValueError(
                "training_target must be exact_sample or independent_resampling"
            )
        self.size = size
        self.num_points = num_points
        self.seed = seed
        self.split = split
        self.training_target = training_target
        self.boundary_band = boundary_band

    def __len__(self) -> int:
        return self.size

    def triplet(self, index: int) -> ProceduralSurfaceTriplet:
        return generate_surface_triplet(
            index,
            num_points=self.num_points,
            seed=self.seed,
            split=self.split,
            boundary_band=self.boundary_band,
        )

    def surface(self, index: int) -> ProceduralSurface:
        return generate_procedural_surface(index, seed=self.seed, split=self.split)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        triplet = self.triplet(index)
        target = triplet.x_a if self.training_target == "exact_sample" else triplet.x_b
        return {
            "source_points": torch.from_numpy(triplet.x_a.points.copy()),
            "source_normals": torch.from_numpy(triplet.x_a.normals.copy()),
            "target_points": torch.from_numpy(target.points.copy()),
            "target_normals": torch.from_numpy(target.normals.copy()),
            "independent_points": torch.from_numpy(triplet.x_b.points.copy()),
            "independent_normals": torch.from_numpy(triplet.x_b.normals.copy()),
            "fresh_points": torch.from_numpy(triplet.x_c.points.copy()),
            "fresh_normals": torch.from_numpy(triplet.x_c.normals.copy()),
            "fresh_boundary_mask": torch.from_numpy(triplet.x_c.boundary_mask.copy()),
            "fresh_thin_structure_mask": torch.from_numpy(
                triplet.x_c.thin_structure_mask.copy()
            ),
            "family": triplet.surface.family,
            "sample_id": triplet.surface.sample_id,
        }
