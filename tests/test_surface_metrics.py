from __future__ import annotations

import numpy as np
import pytest

from pointconstellation.data import TriangleMesh
from pointconstellation.surface_metrics import (
    mesh_surface_metrics,
    point_to_mesh_distances,
    point_to_triangle_squared_distances,
)


def _unit_triangle_mesh() -> TriangleMesh:
    return TriangleMesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )


def test_point_to_triangle_distance_matches_dense_brute_force_fixture() -> None:
    mesh = _unit_triangle_mesh()
    points = np.asarray(
        [
            [0.2, 0.3, 0.7],
            [0.8, 0.8, 0.2],
            [-0.2, 0.4, -0.3],
            [1.2, -0.1, 0.4],
        ]
    )
    exact = np.sqrt(
        point_to_triangle_squared_distances(points, mesh.vertices[mesh.faces])[:, 0]
    )
    grid = np.linspace(0.0, 1.0, 501)
    first, second = np.meshgrid(grid, grid, indexing="ij")
    mask = first + second <= 1.0
    brute_points = np.column_stack((first[mask], second[mask], np.zeros(mask.sum())))
    brute = np.sqrt(
        np.min(
            np.sum((points[:, None, :] - brute_points[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    )

    assert np.all(exact <= brute + 1e-12)
    assert np.allclose(exact, brute, atol=2e-3)
    chunked, indices = point_to_mesh_distances(
        points, mesh, point_chunk_size=2, triangle_chunk_size=1
    )
    assert np.allclose(chunked, exact)
    assert indices.tolist() == [0, 0, 0, 0]


def test_planar_mesh_surface_rmse_and_normal_consistency() -> None:
    mesh = TriangleMesh(
        vertices=np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    axis = np.linspace(-0.8, 0.8, 5)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

    metrics = mesh_surface_metrics(
        points,
        mesh,
        point_chunk_size=7,
        triangle_chunk_size=1,
        normal_neighbors=6,
    )

    assert metrics["surface_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["normal_consistency"] == pytest.approx(1.0, abs=1e-12)
