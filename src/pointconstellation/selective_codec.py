"""Decode a selective coordinate message through a shared frozen decoder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.bitstream import (
    SelectivePacket,
    decode_selective,
    encode_selective,
)

FloatArray = NDArray[np.float64]
Decoder = Callable[[FloatArray, int], ArrayLike]


def _point_array(points: ArrayLike, *, name: str, allow_empty: bool) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or (not allow_empty and not len(values))
    ):
        suffix = "" if allow_empty else " with N > 0"
        raise ValueError(f"{name} must have shape (N, 3){suffix}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return values


@dataclass(frozen=True)
class SelectiveDecodeResult:
    """Decoded union and an explicit raw-preservation contract check."""

    packet: SelectivePacket
    reconstruction: FloatArray
    decoder_reconstruction: FloatArray
    preservation_error: float


def encode_selective_message(
    constellation_coordinates: ArrayLike,
    preserved_coordinates: ArrayLike,
    *,
    bits: int,
    output_points: int,
    entropy: bool = False,
) -> bytes:
    """Serialize the complete per-cloud selective coordinate message."""

    return encode_selective(
        constellation_coordinates,
        preserved_coordinates,
        bits=bits,
        output_points=output_points,
        entropy=entropy,
    )


def decode_selective_message(
    stream: bytes, *, decoder: Decoder | None
) -> SelectiveDecodeResult:
    """Return ``G(Z_constellation) union Z_preserved`` from serialized values."""

    packet = decode_selective(stream)
    if packet.k1:
        if decoder is None:
            raise ValueError("a nonempty constellation requires a decoder")
        decoded = _point_array(
            decoder(packet.constellation_coordinates, packet.output_points),
            name="decoder reconstruction",
            allow_empty=False,
        )
    else:
        decoded = np.empty((0, 3), dtype=np.float64)
    reconstruction = np.concatenate((decoded, packet.preserved_coordinates), axis=0)
    emitted = reconstruction[-packet.k2 :] if packet.k2 else reconstruction[:0]
    preservation_error = (
        float(np.max(np.abs(emitted - packet.preserved_coordinates)))
        if packet.k2
        else 0.0
    )
    if preservation_error != 0.0 or not np.array_equal(
        emitted, packet.preserved_coordinates
    ):
        raise RuntimeError("selective decoder changed a passed-through lattice point")
    return SelectiveDecodeResult(
        packet=packet,
        reconstruction=reconstruction,
        decoder_reconstruction=decoded,
        preservation_error=preservation_error,
    )
