"""Small NumPy geometry metrics for baseline experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _points(points: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("point cloud must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all():
        raise ValueError("point cloud must contain only finite coordinates")
    return array


def _nearest_squared_distances(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    chunk_size: int = 4096,
) -> NDArray[np.float64]:
    nearest = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk_size):
        chunk = source[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - target[None, :, :]) ** 2, axis=2)
        nearest[start : start + len(chunk)] = squared.min(axis=1)
    return nearest


def point_set_error_metrics(
    first: ArrayLike,
    second: ArrayLike,
    *,
    chunk_size: int = 4096,
) -> dict[str, float]:
    """Return symmetric sample-distance and tail metrics for one cloud pair.

    These distances are to the two finite point samples.  They are not
    distances to an underlying continuous surface.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    first_points = _points(first)
    second_points = _points(second)
    forward = _nearest_squared_distances(
        first_points, second_points, chunk_size=chunk_size
    )
    backward = _nearest_squared_distances(
        second_points, first_points, chunk_size=chunk_size
    )
    directed = np.sqrt(np.concatenate((forward, backward)))
    chamfer_mse = 0.5 * (float(forward.mean()) + float(backward.mean()))
    return {
        "chamfer_mse": chamfer_mse,
        "chamfer_rmse": float(np.sqrt(chamfer_mse)),
        "p90_euclidean": float(np.quantile(directed, 0.90)),
        "p99_euclidean": float(np.quantile(directed, 0.99)),
        "hausdorff": float(directed.max()),
    }


def point_to_plane_mse(
    reconstruction: ArrayLike,
    target: ArrayLike,
    target_normals: ArrayLike,
    *,
    chunk_size: int = 4096,
) -> float:
    """Return symmetric point-to-plane MSE using target-attached normals."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    reconstruction_points = _points(reconstruction)
    target_points = _points(target)
    normals = np.asarray(target_normals, dtype=np.float64)
    if normals.shape != target_points.shape or not np.isfinite(normals).all():
        raise ValueError("target normals must be finite and match target shape")
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("target normals must be nonzero")
    normals = normals / lengths

    forward_values: list[NDArray[np.float64]] = []
    for start in range(0, len(reconstruction_points), chunk_size):
        chunk = reconstruction_points[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - target_points[None, :, :]) ** 2, axis=2)
        indices = squared.argmin(axis=1)
        delta = chunk - target_points[indices]
        forward_values.append(np.sum(delta * normals[indices], axis=1) ** 2)
    forward = np.concatenate(forward_values)

    backward_values: list[NDArray[np.float64]] = []
    for start in range(0, len(target_points), chunk_size):
        chunk = target_points[start : start + chunk_size]
        chunk_normals = normals[start : start + len(chunk)]
        squared = np.sum(
            (chunk[:, None, :] - reconstruction_points[None, :, :]) ** 2,
            axis=2,
        )
        indices = squared.argmin(axis=1)
        delta = chunk - reconstruction_points[indices]
        backward_values.append(np.sum(delta * chunk_normals, axis=1) ** 2)
    backward = np.concatenate(backward_values)
    return float(max(forward.mean(), backward.mean()))


def chamfer_rmse(first: ArrayLike, second: ArrayLike) -> float:
    """Return symmetric root-mean-square Chamfer distance.

    This implementation averages the two directed mean squared distances before
    taking the square root. It is intended for small reproducible baselines,
    not high-throughput training.
    """

    return point_set_error_metrics(first, second)["chamfer_rmse"]


def hausdorff_distance(first: ArrayLike, second: ArrayLike) -> float:
    """Return the symmetric Euclidean Hausdorff distance."""

    return point_set_error_metrics(first, second)["hausdorff"]
