"""Exact continuous-mesh metrics with bounded NumPy working sets."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.data.mesh import TriangleMesh


def _point_array(points: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("points must have shape (N, 3) with N > 0")
    if not np.isfinite(values).all():
        raise ValueError("points must contain only finite values")
    return values


def _triangle_array(triangles: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(triangles, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not len(values):
        raise ValueError("triangles must have shape (T, 3, 3) with T > 0")
    if not np.isfinite(values).all():
        raise ValueError("triangles must contain only finite values")
    cross = np.cross(values[:, 1] - values[:, 0], values[:, 2] - values[:, 0])
    if np.any(np.sum(cross * cross, axis=1) <= 1e-24):
        raise ValueError("triangles must be non-degenerate")
    return values


def point_to_triangle_squared_distances(
    points: ArrayLike, triangles: ArrayLike
) -> NDArray[np.float64]:
    """Return every exact point-to-triangle squared distance as an ``N x T`` array."""

    point_values = _point_array(points)
    triangle_values = _triangle_array(triangles)
    point = point_values[:, None, :]
    first = triangle_values[None, :, 0]
    second = triangle_values[None, :, 1]
    third = triangle_values[None, :, 2]
    first_second = second - first
    first_third = third - first
    normal = np.cross(first_second, first_third)
    normal_squared = np.sum(normal * normal, axis=2)

    first_point = point - first
    plane_fraction = np.sum(first_point * normal, axis=2) / normal_squared
    projected = point - plane_fraction[:, :, None] * normal
    projected_delta = projected - first
    dot_00 = np.sum(first_second * first_second, axis=2)
    dot_01 = np.sum(first_second * first_third, axis=2)
    dot_11 = np.sum(first_third * first_third, axis=2)
    dot_20 = np.sum(projected_delta * first_second, axis=2)
    dot_21 = np.sum(projected_delta * first_third, axis=2)
    denominator = dot_00 * dot_11 - dot_01 * dot_01
    barycentric_second = (dot_11 * dot_20 - dot_01 * dot_21) / denominator
    barycentric_third = (dot_00 * dot_21 - dot_01 * dot_20) / denominator
    inside = (
        (barycentric_second >= -1e-12)
        & (barycentric_third >= -1e-12)
        & (barycentric_second + barycentric_third <= 1.0 + 1e-12)
    )
    plane_squared = np.sum((point - projected) ** 2, axis=2)
    plane_squared = np.where(inside, plane_squared, np.inf)

    def segment_squared(
        start: NDArray[np.float64], end: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        edge = end - start
        fraction = np.sum((point - start) * edge, axis=2) / np.sum(edge * edge, axis=2)
        fraction = np.clip(fraction, 0.0, 1.0)
        closest = start + fraction[:, :, None] * edge
        return np.sum((point - closest) ** 2, axis=2)

    return np.minimum.reduce(
        (
            plane_squared,
            segment_squared(first, second),
            segment_squared(second, third),
            segment_squared(third, first),
        )
    )


def point_to_mesh_distances(
    points: ArrayLike,
    mesh: TriangleMesh,
    *,
    point_chunk_size: int = 256,
    triangle_chunk_size: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return exact closest-triangle distances and indices with bounded chunks."""

    if point_chunk_size < 1 or triangle_chunk_size < 1:
        raise ValueError("mesh metric chunk sizes must be positive")
    point_values = _point_array(points)
    raw_triangles = np.asarray(mesh.vertices[mesh.faces], dtype=np.float64)
    if not np.isfinite(raw_triangles).all():
        raise ValueError("mesh triangles must contain only finite values")
    cross = np.cross(
        raw_triangles[:, 1] - raw_triangles[:, 0],
        raw_triangles[:, 2] - raw_triangles[:, 0],
    )
    valid = np.sum(cross * cross, axis=1) > 1e-24
    if not valid.any():
        raise ValueError("mesh has no non-degenerate triangles")
    original_indices = np.flatnonzero(valid)
    triangles = raw_triangles[valid]
    distances = np.full(len(point_values), np.inf, dtype=np.float64)
    indices = np.full(len(point_values), -1, dtype=np.int64)
    for point_start in range(0, len(point_values), point_chunk_size):
        point_chunk = point_values[point_start : point_start + point_chunk_size]
        chunk_best = np.full(len(point_chunk), np.inf, dtype=np.float64)
        chunk_indices = np.full(len(point_chunk), -1, dtype=np.int64)
        for triangle_start in range(0, len(triangles), triangle_chunk_size):
            triangle_chunk = triangles[
                triangle_start : triangle_start + triangle_chunk_size
            ]
            squared = point_to_triangle_squared_distances(point_chunk, triangle_chunk)
            local_indices = squared.argmin(axis=1)
            local_best = squared[np.arange(len(point_chunk)), local_indices]
            improved = local_best < chunk_best
            chunk_best[improved] = local_best[improved]
            chunk_indices[improved] = original_indices[
                triangle_start + local_indices[improved]
            ]
        stop = point_start + len(point_chunk)
        distances[point_start:stop] = np.sqrt(np.maximum(chunk_best, 0.0))
        indices[point_start:stop] = chunk_indices
    return distances, indices


def estimate_point_normals(
    points: ArrayLike,
    *,
    neighbors: int = 12,
    chunk_size: int = 256,
) -> NDArray[np.float64]:
    """Estimate unoriented point normals with deterministic local PCA."""

    values = _point_array(points)
    if len(values) < 4:
        raise ValueError("normal estimation requires at least four points")
    if neighbors < 3 or chunk_size < 1:
        raise ValueError("neighbors must be at least three and chunk size positive")
    neighbor_count = min(neighbors, len(values) - 1)
    normals = np.empty_like(values)
    for start in range(0, len(values), chunk_size):
        chunk = values[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - values[None, :, :]) ** 2, axis=2)
        rows = np.arange(len(chunk))
        squared[rows, start + rows] = np.inf
        nearest = np.argpartition(squared, neighbor_count - 1, axis=1)[
            :, :neighbor_count
        ]
        neighborhoods = values[nearest]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered)
        _, eigenvectors = np.linalg.eigh(covariance)
        normals[start : start + len(chunk)] = eigenvectors[:, :, 0]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / lengths.clip(min=1e-12)


def mesh_surface_metrics(
    reconstruction: ArrayLike,
    mesh: TriangleMesh,
    *,
    point_chunk_size: int = 256,
    triangle_chunk_size: int = 256,
    normal_neighbors: int = 12,
) -> dict[str, float]:
    """Return reconstruction-to-mesh RMSE and unoriented normal consistency."""

    points = _point_array(reconstruction)
    distances, face_indices = point_to_mesh_distances(
        points,
        mesh,
        point_chunk_size=point_chunk_size,
        triangle_chunk_size=triangle_chunk_size,
    )
    triangles = np.asarray(mesh.vertices[mesh.faces], dtype=np.float64)
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True).clip(min=1e-12)
    estimated = estimate_point_normals(
        points,
        neighbors=normal_neighbors,
        chunk_size=point_chunk_size,
    )
    consistency = np.abs(np.sum(estimated * face_normals[face_indices], axis=1))
    return {
        "surface_rmse": float(np.sqrt(np.mean(distances**2))),
        "normal_consistency": float(np.mean(consistency)),
    }
