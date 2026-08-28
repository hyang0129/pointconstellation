"""A small, deterministic fixed-width constellation bitstream.

The stream is intentionally simple: coordinates are quantized on the declared
``[-1, 1]`` lattice, canonically ordered, and bit-packed without entropy coding.
That makes the reported byte count real and reproducible while keeping the toy
benchmark independent of a platform-specific codec implementation.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAGIC = b"PCON"
VERSION = 1
DOMAIN_UNIT_CUBE = 1
HEADER = struct.Struct(">4sBBBBHI")
MODE_IDS = {"free": 1, "strict_subset": 2, "fps": 3}
ID_MODES = {value: key for key, value in MODE_IDS.items()}


@dataclass(frozen=True)
class ConstellationPacket:
    """Decoded coordinate payload and the metadata required to interpret it."""

    coordinates: NDArray[np.float64]
    bits: int
    mode: str
    output_points: int
    payload_bits: int
    header_bytes: int
    payload_bytes: int
    stream_bytes: int


def expected_payload_bytes(constellation_size: int, bits: int) -> int:
    """Return the byte-aligned coordinate payload size without the header."""

    if not 1 <= constellation_size <= 65535:
        raise ValueError("constellation_size must be between 1 and 65535")
    if not 2 <= bits <= 24:
        raise ValueError("bits must be between 2 and 24")
    return math.ceil(3 * constellation_size * bits / 8)


def expected_stream_bytes(constellation_size: int, bits: int) -> int:
    """Return the exact fixed-width stream size, including the header."""

    return HEADER.size + expected_payload_bytes(constellation_size, bits)


def _coordinates(points: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("coordinates must have shape (K, 3) with K > 0")
    if not np.isfinite(array).all():
        raise ValueError("coordinates must be finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("coordinates must lie in the declared [-1, 1] domain")
    return array


def _pack(values: NDArray[np.uint32], bits: int) -> bytes:
    output = bytearray()
    accumulator = 0
    buffered = 0
    for value in values:
        accumulator = (accumulator << bits) | int(value)
        buffered += bits
        while buffered >= 8:
            buffered -= 8
            output.append((accumulator >> buffered) & 0xFF)
            accumulator &= (1 << buffered) - 1 if buffered else 0
    if buffered:
        output.append((accumulator << (8 - buffered)) & 0xFF)
    return bytes(output)


def _unpack(payload: bytes, count: int, bits: int) -> NDArray[np.uint32]:
    values = np.empty(count, dtype=np.uint32)
    accumulator = 0
    buffered = 0
    offset = 0
    for index in range(count):
        while buffered < bits:
            if offset >= len(payload):
                raise ValueError("truncated constellation payload")
            accumulator = (accumulator << 8) | payload[offset]
            buffered += 8
            offset += 1
        buffered -= bits
        values[index] = (accumulator >> buffered) & ((1 << bits) - 1)
        accumulator &= (1 << buffered) - 1 if buffered else 0
    if buffered and accumulator:
        raise ValueError("non-zero constellation padding bits")
    if any(payload[offset:]):
        raise ValueError("unexpected non-zero bytes after constellation payload")
    return values


def encode_constellation(
    coordinates: ArrayLike,
    *,
    bits: int,
    mode: str,
    output_points: int,
) -> bytes:
    """Encode an unordered ``K x 3`` coordinate set into a canonical stream."""

    points = _coordinates(coordinates)
    constellation_size = len(points)
    expected_stream_bytes(constellation_size, bits)
    if mode not in MODE_IDS:
        raise ValueError(f"unknown constellation mode: {mode}")
    if not constellation_size <= output_points <= 0xFFFFFFFF:
        raise ValueError("output_points must fit the stream and be at least K")

    levels = (1 << bits) - 1
    lattice = np.rint((points + 1.0) * 0.5 * levels).astype(np.uint32)
    order = np.lexsort((lattice[:, 2], lattice[:, 1], lattice[:, 0]))
    lattice = lattice[order]
    header = HEADER.pack(
        MAGIC,
        VERSION,
        MODE_IDS[mode],
        bits,
        DOMAIN_UNIT_CUBE,
        constellation_size,
        output_points,
    )
    return header + _pack(lattice.reshape(-1), bits)


def decode_constellation(stream: bytes) -> ConstellationPacket:
    """Decode and validate a fixed-width constellation stream."""

    if len(stream) < HEADER.size:
        raise ValueError("truncated constellation header")
    magic, version, mode_id, bits, domain, size, output_points = HEADER.unpack_from(
        stream
    )
    if magic != MAGIC:
        raise ValueError("invalid constellation magic")
    if version != VERSION:
        raise ValueError(f"unsupported constellation version: {version}")
    if mode_id not in ID_MODES:
        raise ValueError(f"unknown constellation mode id: {mode_id}")
    if domain != DOMAIN_UNIT_CUBE:
        raise ValueError(f"unsupported coordinate domain id: {domain}")
    expected = expected_stream_bytes(size, bits)
    if len(stream) != expected:
        raise ValueError(f"stream has {len(stream)} bytes; expected {expected}")
    if output_points < size:
        raise ValueError("stream output point count is smaller than K")

    values = _unpack(stream[HEADER.size :], 3 * size, bits).reshape(size, 3)
    levels = (1 << bits) - 1
    coordinates = values.astype(np.float64) * (2.0 / levels) - 1.0
    return ConstellationPacket(
        coordinates=coordinates,
        bits=bits,
        mode=ID_MODES[mode_id],
        output_points=output_points,
        payload_bits=3 * size * bits,
        header_bytes=HEADER.size,
        payload_bytes=len(stream) - HEADER.size,
        stream_bytes=len(stream),
    )
