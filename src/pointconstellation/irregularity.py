"""Source-only local irregularity scores and deterministic spaced selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _points(points: ArrayLike, *, name: str = "points") -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError(f"{name} must have shape (N, 3) with N > 0")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return values


def _score_array(scores: ArrayLike, count: int) -> FloatArray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("scores must contain one finite value per point")
    return values


@dataclass(frozen=True)
class LocalGeometry:
    """Reusable deterministic k-NN PCA quantities for one source cloud."""

    curvature: FloatArray
    density_deviation: FloatArray
    boundary: FloatArray
    thin_structure: FloatArray
    normals: FloatArray
    neighbor_indices: IntArray


def _coordinate_tie_ranks(points: FloatArray) -> IntArray:
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ranks = np.empty(len(points), dtype=np.int64)
    ranks[order] = np.arange(len(points))
    return ranks


def local_geometry_scores(
    points: ArrayLike, *, neighbors: int = 16, chunk_size: int = 256
) -> LocalGeometry:
    """Compute PCA curvature, density, boundary, and thinness proxies.

    The boundary score combines one-sided neighborhood displacement with
    unoriented normal disagreement.  It is the finite-cloud source-only proxy
    used for Experiment 040; it is not an analytic mesh-boundary label.
    """

    values = _points(points)
    if not 3 <= neighbors < len(values):
        raise ValueError("neighbors must be in [3, number of points)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    squared_norms = np.einsum("ij,ij->i", values, values)
    tie_ranks = _coordinate_tie_ranks(values)
    indices = np.empty((len(values), neighbors), dtype=np.int64)
    eigenvalues = np.empty((len(values), 3), dtype=np.float64)
    normals = np.empty_like(values)
    density_radius = np.empty(len(values), dtype=np.float64)
    centroid_offset = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        query = values[start:stop]
        squared = (
            np.einsum("ij,ij->i", query, query)[:, None]
            + squared_norms[None, :]
            - 2.0 * query @ values.T
        ).clip(min=0.0)
        squared[np.arange(stop - start), np.arange(start, stop)] = np.inf
        scale = np.maximum(
            1.0, np.max(np.where(np.isfinite(squared), squared, 0.0), axis=1)
        )
        tie_epsilon = (
            np.finfo(np.float64).eps
            * scale[:, None]
            * tie_ranks[None, :]
            / max(len(values), 1)
        )
        nearest = np.argpartition(squared + tie_epsilon, neighbors - 1, axis=1)[
            :, :neighbors
        ]
        nearest_squared = np.take_along_axis(squared, nearest, axis=1)
        neighbor_ranks = tie_ranks[nearest]
        for row in range(len(nearest)):
            order = np.lexsort((neighbor_ranks[row], nearest_squared[row]))
            nearest[row] = nearest[row, order]
            nearest_squared[row] = nearest_squared[row, order]
        neighborhoods = values[nearest]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbors
        local_values, local_vectors = np.linalg.eigh(covariance)
        indices[start:stop] = nearest
        eigenvalues[start:stop] = local_values
        normals[start:stop] = local_vectors[:, :, 0]
        distances = np.sqrt(nearest_squared.clip(min=0.0))
        density_radius[start:stop] = distances.mean(axis=1)
        offset = np.linalg.norm(
            query - neighborhoods.mean(axis=1), axis=1
        ) / density_radius[start:stop].clip(min=1e-12)
        centroid_offset[start:stop] = offset

    axes = np.abs(normals).argmax(axis=1)
    normals[normals[np.arange(len(normals)), axes] < 0] *= -1.0
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    neighbor_normals = normals[indices]
    normal_disagreement = 1.0 - np.mean(
        np.abs(np.einsum("nij,nj->ni", neighbor_normals, normals)), axis=1
    )
    total_variation = eigenvalues.sum(axis=1).clip(min=1e-18)
    curvature = (eigenvalues[:, 0] / total_variation).clip(0.0, 1.0)
    positive_radius = density_radius[density_radius > 1e-15]
    reference_radius = (
        float(np.median(positive_radius)) if len(positive_radius) else 1.0
    )
    density_deviation = np.abs(
        np.log(density_radius.clip(min=1e-15) / reference_radius)
    )
    boundary = 0.5 * centroid_offset + 0.5 * normal_disagreement
    thin_structure = 1.0 - (
        (eigenvalues[:, 1] - eigenvalues[:, 0]) / eigenvalues[:, 2].clip(min=1e-18)
    )
    return LocalGeometry(
        curvature=curvature,
        density_deviation=density_deviation,
        boundary=boundary.clip(0.0),
        thin_structure=thin_structure.clip(0.0, 1.0),
        normals=normals,
        neighbor_indices=indices,
    )


def local_pca_curvature(
    points: ArrayLike, *, neighbors: int = 16, chunk_size: int = 256
) -> FloatArray:
    """Return local surface variation (smallest PCA eigenvalue / trace)."""

    return local_geometry_scores(
        points, neighbors=neighbors, chunk_size=chunk_size
    ).curvature


def local_density_deviation(
    points: ArrayLike, *, neighbors: int = 16, chunk_size: int = 256
) -> FloatArray:
    """Return absolute log deviation of local mean k-NN radius from its median."""

    return local_geometry_scores(
        points, neighbors=neighbors, chunk_size=chunk_size
    ).density_deviation


def boundary_score(
    points: ArrayLike, *, neighbors: int = 16, chunk_size: int = 256
) -> FloatArray:
    """Return the source-only finite-sample boundary/sharp-feature proxy."""

    return local_geometry_scores(
        points, neighbors=neighbors, chunk_size=chunk_size
    ).boundary


def nearest_distances(
    query: ArrayLike, reference: ArrayLike, *, chunk_size: int = 256
) -> FloatArray:
    """Return deterministic one-way Euclidean nearest-neighbor distances."""

    first = _points(query, name="query")
    second = _points(reference, name="reference")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result = np.empty(len(first), dtype=np.float64)
    second_norms = np.einsum("ij,ij->i", second, second)
    for start in range(0, len(first), chunk_size):
        stop = min(start + chunk_size, len(first))
        part = first[start:stop]
        squared = (
            np.einsum("ij,ij->i", part, part)[:, None]
            + second_norms[None, :]
            - 2.0 * part @ second.T
        ).clip(min=0.0)
        result[start:stop] = np.sqrt(squared.min(axis=1))
    return result


def decoder_residual_score(
    source_points: ArrayLike,
    reconstruction: ArrayLike,
    *,
    chunk_size: int = 256,
) -> FloatArray:
    """Return source-to-constellation-decode residuals for source-only selection."""

    return nearest_distances(source_points, reconstruction, chunk_size=chunk_size)


def deterministic_random_scores(points: ArrayLike, *, seed: int) -> FloatArray:
    """Return coordinate-keyed pseudorandom scores independent of point order."""

    values = _points(points)
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    prefix = seed.to_bytes(8, "big", signed=False)
    scores = np.empty(len(values), dtype=np.float64)
    for index, point in enumerate(values):
        digest = hashlib.sha256(prefix + point.astype(">f8").tobytes()).digest()
        scores[index] = int.from_bytes(digest[:8], "big") / 2**64
    return scores


def select_spaced_indices(
    points: ArrayLike,
    scores: ArrayLike,
    count: int,
    *,
    minimum_spacing: float,
) -> IntArray:
    """Greedily select top-scored points under an exact spacing constraint."""

    values = _points(points)
    score_values = _score_array(scores, len(values))
    if not 0 <= count <= len(values):
        raise ValueError("count must be between zero and N")
    if minimum_spacing < 0 or not np.isfinite(minimum_spacing):
        raise ValueError("minimum_spacing must be finite and nonnegative")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0], -score_values))
    selected: list[int] = []
    threshold = minimum_spacing**2
    for raw_index in order:
        index = int(raw_index)
        if selected:
            squared = np.sum((values[selected] - values[index]) ** 2, axis=1)
            if np.any(squared < threshold):
                continue
        selected.append(index)
        if len(selected) == count:
            return np.asarray(selected, dtype=np.int64)
    raise ValueError(
        f"minimum spacing {minimum_spacing} permits only {len(selected)} of "
        f"the requested {count} points"
    )


def stratification_bins(
    scores: ArrayLike, *, bins: int = 5, points: ArrayLike | None = None
) -> IntArray:
    """Assign every score to a deterministic equal-count quantile bin."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite vector")
    if bins < 2:
        raise ValueError("bins must be at least two")
    if points is None:
        order = np.argsort(values, kind="mergesort")
    else:
        coordinates = _points(points)
        if len(coordinates) != len(values):
            raise ValueError("stratification points and scores must align")
        order = np.lexsort(
            (coordinates[:, 2], coordinates[:, 1], coordinates[:, 0], values)
        )
    result = np.empty(len(values), dtype=np.int64)
    result[order] = np.minimum(
        bins - 1, np.arange(len(values), dtype=np.int64) * bins // len(values)
    )
    return result


def stratified_error(
    errors: ArrayLike, assignments: ArrayLike, *, bins: int = 5
) -> list[dict[str, float | int | None]]:
    """Summarize per-point Euclidean errors in precomputed strata."""

    values = np.asarray(errors, dtype=np.float64)
    groups = np.asarray(assignments, dtype=np.int64)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("errors and assignments must be aligned vectors")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("errors must be finite and nonnegative")
    if np.any(groups < 0) or np.any(groups >= bins):
        raise ValueError("assignments are outside the declared bins")
    result: list[dict[str, float | int | None]] = []
    for index in range(bins):
        selected = values[groups == index]
        result.append(
            {
                "quintile": index + 1,
                "count": len(selected),
                "mean_euclidean": float(selected.mean()) if len(selected) else None,
                "mse": float(np.mean(selected**2)) if len(selected) else None,
                "rmse": float(np.sqrt(np.mean(selected**2))) if len(selected) else None,
            }
        )
    if sum(int(row["count"]) for row in result) != len(values):
        raise RuntimeError("stratification bins do not cover the target cloud")
    return result
