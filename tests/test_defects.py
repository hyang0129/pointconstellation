"""Tests for deterministic Experiment 041 synthetic defects."""

from __future__ import annotations

import numpy as np
import pytest

from pointconstellation.defects import (
    DEFECT_TYPES,
    defect_seed,
    inject_defect,
    inject_defect_for_cloud,
    transfer_point_labels,
)


def _sphere(count: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    z = rng.uniform(-1.0, 1.0, count)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = np.sqrt(1.0 - z * z)
    normals = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    return (0.7 * normals).astype(np.float32), normals.astype(np.float32)


@pytest.mark.parametrize("defect_type", DEFECT_TYPES)
def test_defect_injection_is_deterministic_and_labels_declared_count(
    defect_type: str,
) -> None:
    points, normals = _sphere()
    first = inject_defect(
        points,
        defect_type,
        seed=31,
        fraction=0.03,
        normals=normals,
    )
    repeated = inject_defect(
        points,
        defect_type,
        seed=31,
        fraction=0.03,
        normals=normals,
    )

    assert np.array_equal(first.points, repeated.points)
    assert np.array_equal(first.point_labels, repeated.point_labels)
    assert np.array_equal(first.source_indices, repeated.source_indices)
    assert first.defective_count == round(0.03 * len(points))
    assert int(first.point_labels.sum()) == first.defective_count
    assert 0.01 <= first.declared_fraction <= 0.05
    assert first.cloud_label == 1


def test_hole_reduces_cardinality_by_declared_count() -> None:
    points, normals = _sphere(400)
    result = inject_defect(
        points,
        "hole",
        seed=43,
        fraction=0.05,
        normals=normals,
    )

    assert result.removed_count == 20
    assert len(result.points) == len(points) - 20
    assert len(result.point_labels) == len(result.points)
    assert int(result.point_labels.sum()) == 20
    assert np.all(result.source_indices >= 0)


def test_cloud_seed_is_stable_and_traversal_order_independent() -> None:
    points, normals = _sphere(200)
    first = inject_defect_for_cloud(
        points,
        "bump",
        base_seed=53,
        cloud_id="chair:chair_0001",
        normals=normals,
    )
    repeated = inject_defect(
        points,
        "bump",
        seed=defect_seed(53, "chair:chair_0001", "bump"),
        normals=normals,
    )

    assert np.array_equal(first.points, repeated.points)
    assert np.array_equal(first.point_labels, repeated.point_labels)
    assert first.requested_fraction == repeated.requested_fraction


@pytest.mark.parametrize("defect_type", DEFECT_TYPES)
def test_defect_geometry_is_input_permutation_equivariant(defect_type: str) -> None:
    points, normals = _sphere(200)
    permutation = np.random.default_rng(61).permutation(len(points))
    first = inject_defect(
        points,
        defect_type,
        seed=67,
        fraction=0.03,
        normals=normals,
    )
    permuted = inject_defect(
        points[permutation],
        defect_type,
        seed=67,
        fraction=0.03,
        normals=normals[permutation],
    )

    first_values = np.column_stack((first.points, first.point_labels))
    permuted_values = np.column_stack((permuted.points, permuted.point_labels))
    first_order = np.lexsort(
        (
            first_values[:, 3],
            first_values[:, 2],
            first_values[:, 1],
            first_values[:, 0],
        )
    )
    permuted_order = np.lexsort(
        (
            permuted_values[:, 3],
            permuted_values[:, 2],
            permuted_values[:, 1],
            permuted_values[:, 0],
        )
    )
    assert np.allclose(
        first_values[first_order], permuted_values[permuted_order], atol=1e-7
    )


def test_undefected_control_is_an_exact_copy() -> None:
    points, _ = _sphere(128)
    result = inject_defect(points, "none", seed=59)

    assert np.array_equal(result.points, points)
    assert not np.shares_memory(result.points, points)
    assert not result.point_labels.any()
    assert result.cloud_label == 0
    assert result.defect_type == "none"


def test_injection_rejects_source_outside_declared_codec_domain() -> None:
    points, normals = _sphere(128)
    points[0, 0] = 1.001

    with pytest.raises(ValueError, match="defect source points.*codec domain"):
        inject_defect(points, "bump", seed=61, normals=normals)


def test_nearest_sample_label_transfer_handles_changed_cardinality() -> None:
    reference = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0], dtype=np.uint8)
    query = np.asarray([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float32)

    transferred = transfer_point_labels(reference, labels, query)

    assert transferred.tolist() == [0, 1]
