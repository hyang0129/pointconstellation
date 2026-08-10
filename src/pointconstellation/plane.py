"""A four-point, geometry-only codec for approximately planar point clouds."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_points(points: ArrayLike, *, minimum: int) -> FloatArray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(array) < minimum:
        raise ValueError(f"at least {minimum} points are required")
    if not np.isfinite(array).all():
        raise ValueError("points must contain only finite coordinates")
    return array


def encode_plane_constellation(
    points: ArrayLike, *, max_rmse: float | None = None
) -> FloatArray:
    """Encode an approximately planar cloud as four coplanar corner points.

    The encoder fits a least-squares plane, finds the axis-aligned rectangle in
    the plane's principal coordinate system, and returns its corners in world
    coordinates. The returned array contains no metadata or latent features.

    Args:
        points: Input array with shape ``(N, 3)``.
        max_rmse: Optional maximum point-to-fitted-plane RMSE. If supplied, a
            cloud that is not planar enough raises ``ValueError``.

    Returns:
        Four 3D points with shape ``(4, 3)``.
    """

    cloud = _as_points(points, minimum=4)
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)

    if singular_values[1] <= np.finfo(np.float64).eps:
        raise ValueError("points do not span a two-dimensional surface")

    basis = axes[:2]
    normal = axes[2]
    rmse = float(np.sqrt(np.mean((centered @ normal) ** 2)))
    if max_rmse is not None and rmse > max_rmse:
        raise ValueError(f"plane RMSE {rmse:.6g} exceeds maximum {max_rmse:.6g}")

    projected = centered @ basis.T
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    uv_corners = np.array(
        [
            [lower[0], lower[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
            [lower[0], upper[1]],
        ]
    )
    return centroid + uv_corners @ basis


def _cyclic_corners(constellation: ArrayLike) -> FloatArray:
    corners = _as_points(constellation, minimum=4)
    if corners.shape != (4, 3):
        raise ValueError("a plane constellation must have shape (4, 3)")

    centroid = corners.mean(axis=0)
    centered = corners - centroid
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if singular_values[1] <= np.finfo(np.float64).eps:
        raise ValueError("constellation corners do not span a plane")

    projected = centered @ axes[:2].T
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    return corners[np.argsort(angles)]


def _radical_inverse(indices: NDArray[np.int64], base: int) -> FloatArray:
    values = np.zeros(len(indices), dtype=np.float64)
    factor = 1.0 / base
    remaining = indices.copy()
    while np.any(remaining):
        values += factor * (remaining % base)
        remaining //= base
        factor /= base
    return values


def _symmetric_uv_samples(num_points: int) -> FloatArray:
    """Sample a square without depending on its corner labels.

    A cyclic corner sort has an arbitrary starting corner and direction. Each
    Halton seed is therefore expanded through the eight symmetries of a square.
    Any leftover samples are placed at the invariant center. This makes the
    decoded point *set* independent of the input array's corner order.
    """

    orbit_count, remainder = divmod(num_points, 8)
    if orbit_count:
        indices = np.arange(1, orbit_count + 1, dtype=np.int64)
        u = _radical_inverse(indices, 2)
        v = _radical_inverse(indices, 3)
        samples = (
            np.stack(
                (
                    np.column_stack((u, v)),
                    np.column_stack((1.0 - u, v)),
                    np.column_stack((u, 1.0 - v)),
                    np.column_stack((1.0 - u, 1.0 - v)),
                    np.column_stack((v, u)),
                    np.column_stack((1.0 - v, u)),
                    np.column_stack((v, 1.0 - u)),
                    np.column_stack((1.0 - v, 1.0 - u)),
                )
            )
            .transpose(1, 0, 2)
            .reshape(-1, 2)
        )
    else:
        samples = np.empty((0, 2), dtype=np.float64)

    if remainder >= 4:
        samples = np.vstack(
            (samples, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
        )
        remainder -= 4
    if remainder:
        samples = np.vstack((samples, np.full((remainder, 2), 0.5)))
    return samples


def decode_plane_constellation(
    constellation: ArrayLike, *, num_points: int
) -> FloatArray:
    """Decode four unordered corners into a dense, deterministic plane sample.

    ``num_points`` is a decoder/rendering configuration rather than information
    hidden in the constellation. Sampling uses a two-dimensional Halton
    sequence, which gives useful coverage for any requested output size.
    """

    if isinstance(num_points, bool) or not isinstance(num_points, (int, np.integer)):
        raise TypeError("num_points must be an integer")
    if num_points < 1:
        raise ValueError("num_points must be positive")

    c0, c1, c2, c3 = _cyclic_corners(constellation)
    samples = _symmetric_uv_samples(num_points)
    u = samples[:, :1]
    v = samples[:, 1:]

    return (
        (1.0 - u) * (1.0 - v) * c0
        + u * (1.0 - v) * c1
        + u * v * c2
        + (1.0 - u) * v * c3
    )


def point_to_constellation_plane_rmse(
    points: ArrayLike, constellation: ArrayLike
) -> float:
    """Return orthogonal RMSE from points to a constellation's fitted plane."""

    cloud = _as_points(points, minimum=1)
    corners = _cyclic_corners(constellation)
    centroid = corners.mean(axis=0)
    _, _, axes = np.linalg.svd(corners - centroid, full_matrices=False)
    distances = (cloud - centroid) @ axes[2]
    return float(np.sqrt(np.mean(distances**2)))
