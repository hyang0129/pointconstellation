"""Generate, encode, and decode a synthetic wall."""

from __future__ import annotations

import argparse
import json

import numpy as np

from pointconstellation.metrics import chamfer_rmse, hausdorff_distance
from pointconstellation.plane import (
    decode_plane_constellation,
    encode_plane_constellation,
    point_to_constellation_plane_rmse,
)


def synthetic_wall(
    num_points: int,
    *,
    width: float = 4.0,
    height: float = 2.5,
    noise: float = 0.005,
    seed: int = 7,
) -> np.ndarray:
    """Return a rotated, translated, and optionally noisy wall point cloud."""

    rng = np.random.default_rng(seed)
    uv = rng.uniform(
        [-width / 2, -height / 2], [width / 2, height / 2], (num_points, 2)
    )
    local = np.column_stack((uv, rng.normal(0.0, noise, num_points)))

    yaw, pitch, roll = np.deg2rad([28.0, -13.0, 7.0])
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    return local @ (rz @ ry @ rx).T + np.array([1.25, -0.8, 2.0])


def run_demo(*, num_points: int, noise: float, seed: int) -> dict[str, float | int]:
    cloud = synthetic_wall(num_points, noise=noise, seed=seed)
    constellation = encode_plane_constellation(cloud)
    reconstruction = decode_plane_constellation(constellation, num_points=num_points)

    input_bytes = int(cloud.astype(np.float32).nbytes)
    constellation_bytes = int(constellation.astype(np.float32).nbytes)
    return {
        "input_points": len(cloud),
        "constellation_points": len(constellation),
        "coordinate_ratio": len(cloud) / len(constellation),
        "float32_input_coordinate_bytes": input_bytes,
        "float32_constellation_coordinate_bytes": constellation_bytes,
        "chamfer_rmse": chamfer_rmse(cloud, reconstruction),
        "hausdorff": hausdorff_distance(cloud, reconstruction),
        "input_to_fitted_plane_rmse": point_to_constellation_plane_rmse(
            cloud, constellation
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.points < 4:
        parser.error("--points must be at least 4")
    if args.noise < 0:
        parser.error("--noise cannot be negative")
    print(
        json.dumps(
            run_demo(num_points=args.points, noise=args.noise, seed=args.seed),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
