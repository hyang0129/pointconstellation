"""Geometry-only point constellation experiments."""

from __future__ import annotations

from typing import Any

__all__ = [
    "chamfer_rmse",
    "decode_plane_constellation",
    "encode_plane_constellation",
    "hausdorff_distance",
    "point_to_constellation_plane_rmse",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Load numerical convenience exports only when they are requested."""

    if name in {"chamfer_rmse", "hausdorff_distance"}:
        from pointconstellation import metrics

        return getattr(metrics, name)
    if name in {
        "decode_plane_constellation",
        "encode_plane_constellation",
        "point_to_constellation_plane_rmse",
    }:
        from pointconstellation import plane

        return getattr(plane, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
