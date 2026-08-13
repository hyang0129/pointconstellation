"""Coordinate quantization for constellation bottlenecks."""

from __future__ import annotations

import torch
from torch import Tensor


def quantization_step(bits: int, *, lower: float = -1.0, upper: float = 1.0) -> float:
    if bits < 2 or bits > 24:
        raise ValueError("bits must be between 2 and 24")
    if upper <= lower:
        raise ValueError("upper must exceed lower")
    return (upper - lower) / ((1 << bits) - 1)


def quantize_coordinates(
    coordinates: Tensor,
    bits: int,
    *,
    lower: float = -1.0,
    upper: float = 1.0,
) -> Tensor:
    """Quantize and dequantize coordinates onto a uniform lattice."""

    step = quantization_step(bits, lower=lower, upper=upper)
    clamped = coordinates.clamp(lower, upper)
    return torch.round((clamped - lower) / step) * step + lower


def quantize_ste(
    coordinates: Tensor,
    bits: int,
    *,
    training: bool,
    jitter: bool = True,
    lower: float = -1.0,
    upper: float = 1.0,
) -> Tensor:
    """Quantize with optional half-bin jitter and a straight-through gradient."""

    step = quantization_step(bits, lower=lower, upper=upper)
    values = coordinates
    if training and jitter:
        values = values + (torch.rand_like(values) - 0.5) * step
    quantized = quantize_coordinates(values, bits, lower=lower, upper=upper)
    return values + (quantized - values).detach()
