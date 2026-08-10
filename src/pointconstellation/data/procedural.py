"""Deterministic procedural surfaces with analytic normals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FAMILIES = ("plane", "corner", "box", "sphere", "cylinder", "beam", "pair")
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class ProceduralSample:
    points: FloatArray
    normals: FloatArray
    family: str
    sample_id: int


def _unit_vectors(rng: np.random.Generator, count: int) -> FloatArray:
    vectors = rng.normal(size=(count, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    return vectors.astype(np.float32)


def _plane(
    rng: np.random.Generator, count: int, *, ood: bool
) -> tuple[FloatArray, FloatArray]:
    width = rng.uniform(1.5, 2.0) if ood else rng.uniform(0.8, 1.5)
    height = rng.uniform(0.2, 0.5) if ood else rng.uniform(0.6, 1.3)
    uv = rng.uniform((-width, -height), (width, height), size=(count, 2))
    points = np.column_stack((uv, np.zeros(count)))
    normals = np.tile((0.0, 0.0, 1.0), (count, 1))
    return points.astype(np.float32), normals.astype(np.float32)


def _corner(
    rng: np.random.Generator, count: int, *, ood: bool
) -> tuple[FloatArray, FloatArray]:
    first = count // 2
    second = count - first
    reach = rng.uniform(1.3, 1.8) if ood else rng.uniform(0.7, 1.2)
    xy = rng.uniform(-reach, reach, size=(first, 2))
    xz = rng.uniform(-reach, reach, size=(second, 2))
    plane_a = np.column_stack((xy, np.zeros(first)))
    plane_b = np.column_stack((xz[:, 0], np.zeros(second), xz[:, 1]))
    points = np.vstack((plane_a, plane_b))
    normals = np.vstack(
        (
            np.tile((0.0, 0.0, 1.0), (first, 1)),
            np.tile((0.0, 1.0, 0.0), (second, 1)),
        )
    )
    return points.astype(np.float32), normals.astype(np.float32)


def _box_surface(
    rng: np.random.Generator,
    count: int,
    half_extents: NDArray[np.float64],
) -> tuple[FloatArray, FloatArray]:
    faces = rng.integers(0, 6, size=count)
    points = rng.uniform(-1.0, 1.0, size=(count, 3)) * half_extents
    normals = np.zeros((count, 3), dtype=np.float64)
    axes = faces // 2
    signs = np.where(faces % 2, 1.0, -1.0)
    rows = np.arange(count)
    points[rows, axes] = signs * half_extents[axes]
    normals[rows, axes] = signs
    return points.astype(np.float32), normals.astype(np.float32)


def _box(
    rng: np.random.Generator, count: int, *, ood: bool
) -> tuple[FloatArray, FloatArray]:
    low, high = (0.2, 1.6) if ood else (0.55, 1.1)
    return _box_surface(rng, count, rng.uniform(low, high, size=3))


def _sphere(
    rng: np.random.Generator, count: int, *, ood: bool
) -> tuple[FloatArray, FloatArray]:
    normals = _unit_vectors(rng, count)
    radius = rng.uniform(1.2, 1.6) if ood else rng.uniform(0.7, 1.1)
    return (normals * radius).astype(np.float32), normals


def _cylinder(
    rng: np.random.Generator, count: int, *, ood: bool
) -> tuple[FloatArray, FloatArray]:
    radius = rng.uniform(0.25, 0.4) if ood else rng.uniform(0.55, 0.9)
    half_height = rng.uniform(1.2, 1.8) if ood else rng.uniform(0.6, 1.1)
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
    return points.astype(np.float32), normals.astype(np.float32)


def _beam(
    rng: np.random.Generator, count: int, *, ood: bool
) -> tuple[FloatArray, FloatArray]:
    thickness = rng.uniform(0.025, 0.08) if ood else rng.uniform(0.08, 0.2)
    extents = np.array([rng.uniform(0.8, 1.4), thickness, rng.uniform(0.4, 0.8)])
    return _box_surface(rng, count, extents)


def _pair(
    rng: np.random.Generator, count: int, *, ood: bool
) -> tuple[FloatArray, FloatArray]:
    first = count // 2
    second = count - first
    normal_a = _unit_vectors(rng, first)
    normal_b = _unit_vectors(rng, second)
    separation = rng.uniform(1.1, 1.5) if ood else rng.uniform(0.65, 1.0)
    radius = rng.uniform(0.2, 0.35)
    points_a = radius * normal_a + np.array([-separation, 0.0, 0.0])
    points_b = radius * normal_b + np.array([separation, 0.0, 0.0])
    return (
        np.vstack((points_a, points_b)).astype(np.float32),
        np.vstack((normal_a, normal_b)).astype(np.float32),
    )


GENERATORS = {
    "plane": _plane,
    "corner": _corner,
    "box": _box,
    "sphere": _sphere,
    "cylinder": _cylinder,
    "beam": _beam,
    "pair": _pair,
}


def _rotation(rng: np.random.Generator) -> NDArray[np.float64]:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1.0
    return matrix


def generate_sample(
    sample_id: int,
    *,
    num_points: int = 1024,
    seed: int = 7,
    split: str = "train",
) -> ProceduralSample:
    """Generate one deterministic normalized procedural surface."""

    if num_points < 8:
        raise ValueError("num_points must be at least 8")
    split_codes = {"train": 0, "validation": 1, "id_test": 2, "parameter_ood": 3}
    if split not in split_codes:
        raise ValueError(f"unknown split: {split}")
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, split_codes[split], sample_id))
    )
    family = FAMILIES[sample_id % len(FAMILIES)]
    points, normals = GENERATORS[family](rng, num_points, ood=split == "parameter_ood")
    rotation = _rotation(rng)
    points = points @ rotation.T
    normals = normals @ rotation.T
    points -= points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max().clip(min=1e-8)
    points = points / scale
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-8)
    permutation = rng.permutation(num_points)
    return ProceduralSample(
        points[permutation].astype(np.float32),
        normals[permutation].astype(np.float32),
        family,
        sample_id,
    )


class ProceduralPointCloudDataset:
    """Lazy PyTorch-compatible dataset generated entirely from sample IDs."""

    def __init__(
        self,
        size: int,
        *,
        num_points: int = 1024,
        seed: int = 7,
        split: str = "train",
    ) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        self.size = size
        self.num_points = num_points
        self.seed = seed
        self.split = split

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        sample = generate_sample(
            index,
            num_points=self.num_points,
            seed=self.seed,
            split=self.split,
        )
        return {
            "points": torch.from_numpy(sample.points.copy()),
            "normals": torch.from_numpy(sample.normals.copy()),
            "family": sample.family,
            "sample_id": sample.sample_id,
        }
