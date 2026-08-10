import itertools

import numpy as np
import pytest

from pointconstellation.metrics import chamfer_rmse, hausdorff_distance
from pointconstellation.plane import (
    decode_plane_constellation,
    encode_plane_constellation,
    point_to_constellation_plane_rmse,
)


def rotated_plane() -> np.ndarray:
    u, v = np.meshgrid(np.linspace(-2.0, 2.0, 25), np.linspace(-1.0, 1.0, 20))
    local = np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size)))
    angle = np.deg2rad(31.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return local @ rotation.T + np.array([3.0, -1.5, 0.75])


def test_plane_encodes_to_four_coordinates_and_decodes_requested_count() -> None:
    cloud = rotated_plane()
    constellation = encode_plane_constellation(cloud, max_rmse=1e-10)
    reconstruction = decode_plane_constellation(constellation, num_points=500)

    assert constellation.shape == (4, 3)
    assert reconstruction.shape == (500, 3)
    assert point_to_constellation_plane_rmse(cloud, constellation) < 1e-12
    assert chamfer_rmse(cloud, reconstruction) < 0.13
    assert hausdorff_distance(cloud, reconstruction) < 0.25


def test_decoder_accepts_unordered_constellation() -> None:
    cloud = rotated_plane()
    constellation = encode_plane_constellation(cloud)
    original = decode_plane_constellation(constellation, num_points=64)

    for permutation in itertools.islice(itertools.permutations(range(4)), 8):
        decoded = decode_plane_constellation(
            constellation[list(permutation)], num_points=64
        )
        assert chamfer_rmse(original, decoded) < 1e-12


def test_encoder_can_reject_non_planar_cloud() -> None:
    rng = np.random.default_rng(3)
    volume = rng.normal(size=(100, 3))

    with pytest.raises(ValueError, match="exceeds maximum"):
        encode_plane_constellation(volume, max_rmse=0.01)


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (np.zeros((3, 3)), "at least 4"),
        (np.zeros((4, 2)), "shape"),
        (np.full((4, 3), np.nan), "finite"),
    ],
)
def test_encoder_validates_input(points: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        encode_plane_constellation(points)


def test_metrics_are_zero_for_identical_clouds() -> None:
    cloud = rotated_plane()[:20]
    assert chamfer_rmse(cloud, cloud) == pytest.approx(0.0)
    assert hausdorff_distance(cloud, cloud) == pytest.approx(0.0)
