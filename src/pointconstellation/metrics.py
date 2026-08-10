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


def chamfer_rmse(first: ArrayLike, second: ArrayLike) -> float:
    """Return symmetric root-mean-square Chamfer distance.

    This implementation averages the two directed mean squared distances before
    taking the square root. It is intended for small reproducible baselines,
    not high-throughput training.
    """

    first_points = _points(first)
    second_points = _points(second)
    forward = _nearest_squared_distances(first_points, second_points).mean()
    backward = _nearest_squared_distances(second_points, first_points).mean()
    return float(np.sqrt(0.5 * (forward + backward)))


def hausdorff_distance(first: ArrayLike, second: ArrayLike) -> float:
    """Return the symmetric Euclidean Hausdorff distance."""

    first_points = _points(first)
    second_points = _points(second)
    forward = _nearest_squared_distances(first_points, second_points).max()
    backward = _nearest_squared_distances(second_points, first_points).max()
    return float(np.sqrt(max(forward, backward)))
