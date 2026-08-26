"""Shared-model and per-object rate accounting utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AMORTIZATION_CORPUS_SIZES = (128, 672, 2468, 10_000, 100_000)


def amortized_bpp(
    per_object_bytes: float,
    model_bytes: int,
    input_points: int,
    corpus_size: int,
) -> float:
    """Return per-object stream plus ``1 / corpus_size`` of a shared model."""

    if per_object_bytes < 0:
        raise ValueError("per_object_bytes cannot be negative")
    if model_bytes < 0:
        raise ValueError("model_bytes cannot be negative")
    if input_points < 1:
        raise ValueError("input_points must be positive")
    if (
        not isinstance(corpus_size, int)
        or isinstance(corpus_size, bool)
        or corpus_size < 1
    ):
        raise ValueError("corpus_size must be a positive integer")
    return 8.0 * (per_object_bytes + model_bytes / corpus_size) / input_points


def amortized_bpp_table(
    per_object_bytes: float,
    model_bytes: int,
    input_points: int,
    *,
    corpus_sizes: tuple[int, ...] = AMORTIZATION_CORPUS_SIZES,
) -> dict[str, float]:
    """Return the predeclared corpus-size accounting points as JSON-ready data."""

    if not corpus_sizes or len(set(corpus_sizes)) != len(corpus_sizes):
        raise ValueError("corpus_sizes must be nonempty and unique")
    return {
        str(size): amortized_bpp(per_object_bytes, model_bytes, input_points, size)
        for size in corpus_sizes
    }


def model_amortization(
    per_object_bytes: float,
    input_points: int,
    model_bytes: Mapping[str, int],
) -> dict[str, Any]:
    """Describe exact model files and their amortized total rates."""

    if not model_bytes:
        raise ValueError("at least one model representation is required")
    sizes = dict(model_bytes)
    if any(not name or value < 0 for name, value in sizes.items()):
        raise ValueError("model byte entries require names and nonnegative sizes")
    return {
        "model_bytes": sizes,
        "amortized_bpp": {
            name: amortized_bpp_table(per_object_bytes, value, input_points)
            for name, value in sizes.items()
        },
    }


def no_model_amortization(per_object_bytes: float, input_points: int) -> dict[str, Any]:
    """Return the common schema for codecs without a shared learned model."""

    return {
        "model_bytes": 0,
        "amortized_bpp": amortized_bpp_table(per_object_bytes, 0, input_points),
    }


def parameter_set_amortized_bpp_table(
    stream_bytes: float,
    parameter_set_bytes: float,
    input_points: int,
    *,
    per_object_side_information_bytes: float = 0.0,
    corpus_sizes: tuple[int, ...] = AMORTIZATION_CORPUS_SIZES,
) -> dict[str, float]:
    """Amortize G-PCC parameter sets while retaining per-object side information."""

    if not 0 <= parameter_set_bytes <= stream_bytes:
        raise ValueError("parameter_set_bytes must lie within the codec stream")
    if per_object_side_information_bytes < 0:
        raise ValueError("per-object side information cannot be negative")
    if input_points < 1:
        raise ValueError("input_points must be positive")
    return {
        str(size): 8.0
        * (
            stream_bytes
            - parameter_set_bytes
            + parameter_set_bytes / size
            + per_object_side_information_bytes
        )
        / input_points
        for size in corpus_sizes
    }
