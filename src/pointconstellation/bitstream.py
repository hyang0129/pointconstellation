"""Deterministic fixed-width and entropy-coded constellation streams.

Both stream modes quantize coordinates on the declared ``[-1, 1]`` lattice
and sort them lexicographically.  Mode 0 is the declared fixed-width stream.
Mode 1 is an optional diagnostic that delta-codes the same unordered lattice
points with a stream-adaptive Rice code.  Neither mode carries learned features
or target-only information.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAGIC = b"PCON"
VERSION = 1
MODE_FIXED = 0
MODE_ENTROPY = 1
STREAM_MODES = (MODE_FIXED, MODE_ENTROPY)
DOMAIN_UNIT_CUBE = 1
HEADER = struct.Struct(">4sBBBBHI")


@dataclass(frozen=True)
class ConstellationPacket:
    """Decoded coordinate payload and the metadata required to interpret it."""

    coordinates: NDArray[np.float64]
    bits: int
    mode: int
    output_points: int
    payload_bits: int
    header_bytes: int
    payload_bytes: int
    stream_bytes: int


class _BitWriter:
    def __init__(self) -> None:
        self._output = bytearray()
        self._accumulator = 0
        self._buffered = 0
        self.bits_written = 0

    def write(self, value: int, width: int) -> None:
        if width < 0 or value < 0 or value >= 1 << width:
            raise ValueError("bit value does not fit its declared width")
        self.bits_written += width
        remaining = width
        while remaining:
            available = 8 - self._buffered
            take = min(remaining, available)
            shift = remaining - take
            self._accumulator = (self._accumulator << take) | (
                (value >> shift) & ((1 << take) - 1)
            )
            self._buffered += take
            remaining -= take
            if self._buffered == 8:
                self._output.append(self._accumulator)
                self._accumulator = 0
                self._buffered = 0

    def write_unary(self, quotient: int) -> None:
        if quotient < 0:
            raise ValueError("Rice quotient cannot be negative")
        while quotient >= 8 and self._buffered == 0:
            self._output.append(0)
            self.bits_written += 8
            quotient -= 8
        for _ in range(quotient):
            self.write(0, 1)
        self.write(1, 1)

    def finish(self) -> bytes:
        if self._buffered:
            self._output.append(self._accumulator << (8 - self._buffered))
            self._accumulator = 0
            self._buffered = 0
        return bytes(self._output)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.position = 0

    def read(self, width: int) -> int:
        if width < 0 or self.position + width > 8 * len(self._data):
            raise ValueError("truncated constellation payload")
        value = 0
        for _ in range(width):
            byte = self._data[self.position // 8]
            shift = 7 - self.position % 8
            value = (value << 1) | ((byte >> shift) & 1)
            self.position += 1
        return value

    def read_unary(self) -> int:
        quotient = 0
        while True:
            if self.position >= 8 * len(self._data):
                raise ValueError("truncated constellation Rice code")
            if self.read(1):
                return quotient
            quotient += 1

    def validate_padding(self) -> None:
        expected_bytes = math.ceil(self.position / 8)
        if len(self._data) != expected_bytes:
            raise ValueError("unexpected bytes after constellation payload")
        while self.position < 8 * len(self._data):
            if self.read(1):
                raise ValueError("non-zero constellation padding bits")


def expected_payload_bytes(constellation_size: int, bits: int) -> int:
    """Return the byte-aligned fixed-width payload size without the header."""

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


def _lattice(coordinates: ArrayLike, bits: int) -> NDArray[np.uint32]:
    points = _coordinates(coordinates)
    expected_stream_bytes(len(points), bits)
    levels = (1 << bits) - 1
    lattice = np.rint((points + 1.0) * 0.5 * levels).astype(np.uint32)
    order = np.lexsort((lattice[:, 2], lattice[:, 1], lattice[:, 0]))
    return lattice[order]


def _pack(values: NDArray[np.uint32], bits: int) -> bytes:
    writer = _BitWriter()
    for value in values:
        writer.write(int(value), bits)
    return writer.finish()


def _unpack(payload: bytes, count: int, bits: int) -> NDArray[np.uint32]:
    reader = _BitReader(payload)
    values = np.asarray([reader.read(bits) for _ in range(count)], dtype=np.uint32)
    reader.validate_padding()
    return values


def _zigzag(deltas: NDArray[np.int64]) -> NDArray[np.uint64]:
    return np.where(deltas >= 0, 2 * deltas, -2 * deltas - 1).astype(np.uint64)


def _rice_parameter(symbols: NDArray[np.uint64], bits: int) -> int:
    if not len(symbols):
        return 0
    costs = []
    for parameter in range(bits + 1):
        quotients = symbols >> parameter
        costs.append(
            sum(int(value) for value in quotients) + len(symbols) * (parameter + 1)
        )
    return min(range(len(costs)), key=costs.__getitem__)


def _pack_entropy(lattice: NDArray[np.uint32], bits: int) -> tuple[bytes, int]:
    deltas = np.diff(lattice.astype(np.int64), axis=0).reshape(-1)
    symbols = _zigzag(deltas)
    parameter = _rice_parameter(symbols, bits)
    writer = _BitWriter()
    for value in lattice[0]:
        writer.write(int(value), bits)
    for symbol_value in symbols:
        symbol = int(symbol_value)
        writer.write_unary(symbol >> parameter)
        if parameter:
            writer.write(symbol & ((1 << parameter) - 1), parameter)
    payload = bytes([parameter]) + writer.finish()
    return payload, 8 + writer.bits_written


def _unpack_entropy(
    payload: bytes, constellation_size: int, bits: int
) -> tuple[NDArray[np.uint32], int]:
    if not payload:
        raise ValueError("truncated constellation Rice parameter")
    parameter = payload[0]
    if parameter > bits:
        raise ValueError(f"invalid constellation Rice parameter: {parameter}")
    reader = _BitReader(payload[1:])
    lattice = np.empty((constellation_size, 3), dtype=np.int64)
    lattice[0] = [reader.read(bits) for _ in range(3)]
    levels = (1 << bits) - 1
    for index in range(1, constellation_size):
        for axis in range(3):
            quotient = reader.read_unary()
            remainder = reader.read(parameter) if parameter else 0
            symbol = (quotient << parameter) | remainder
            delta = symbol // 2 if symbol % 2 == 0 else -(symbol // 2) - 1
            lattice[index, axis] = lattice[index - 1, axis] + delta
        if np.any(lattice[index] < 0) or np.any(lattice[index] > levels):
            raise ValueError("entropy-coded coordinate lies outside the lattice")
        if tuple(lattice[index]) < tuple(lattice[index - 1]):
            raise ValueError("entropy-coded coordinates are not canonical")
    payload_bits = 8 + reader.position
    reader.validate_padding()
    return lattice.astype(np.uint32), payload_bits


def entropy_bound_bytes(coordinates: ArrayLike, *, bits: int) -> float:
    """Return an oracle per-axis order-0 bound for the entropy stream.

    The bound includes the 14-byte header, one-byte Rice parameter, and the
    first point at full precision.  Delta entropy is calculated independently
    for each axis and is not rounded to a byte boundary.
    """

    lattice = _lattice(coordinates, bits)
    deltas = np.diff(lattice.astype(np.int64), axis=0)
    entropy_bits = 0.0
    for axis in range(3):
        if not len(deltas):
            continue
        _, counts = np.unique(deltas[:, axis], return_counts=True)
        probabilities = counts.astype(np.float64) / len(deltas)
        entropy_bits += float(-np.sum(counts * np.log2(probabilities)))
    return HEADER.size + 1 + (3 * bits + entropy_bits) / 8.0


def encode_constellation(
    coordinates: ArrayLike,
    *,
    bits: int,
    mode: int = MODE_FIXED,
    output_points: int,
) -> bytes:
    """Encode an unordered ``K x 3`` coordinate set into a canonical stream."""

    lattice = _lattice(coordinates, bits)
    constellation_size = len(lattice)
    if mode not in STREAM_MODES:
        raise ValueError(f"unknown constellation stream mode: {mode}")
    if not constellation_size <= output_points <= 0xFFFFFFFF:
        raise ValueError("output_points must fit the stream and be at least K")

    header = HEADER.pack(
        MAGIC,
        VERSION,
        mode,
        bits,
        DOMAIN_UNIT_CUBE,
        constellation_size,
        output_points,
    )
    if mode == MODE_FIXED:
        payload = _pack(lattice.reshape(-1), bits)
    else:
        payload, _ = _pack_entropy(lattice, bits)
    return header + payload


def decode_constellation(stream: bytes) -> ConstellationPacket:
    """Decode and validate a fixed-width or entropy-coded constellation stream."""

    if len(stream) < HEADER.size:
        raise ValueError("truncated constellation header")
    magic, version, mode, bits, domain, size, output_points = HEADER.unpack_from(stream)
    if magic != MAGIC:
        raise ValueError("invalid constellation magic")
    if version != VERSION:
        raise ValueError(f"unsupported constellation version: {version}")
    if mode not in STREAM_MODES:
        raise ValueError(f"unknown constellation stream mode: {mode}")
    if domain != DOMAIN_UNIT_CUBE:
        raise ValueError(f"unsupported coordinate domain id: {domain}")
    expected_payload_bytes(size, bits)
    if output_points < size:
        raise ValueError("stream output point count is smaller than K")

    payload = stream[HEADER.size :]
    if mode == MODE_FIXED:
        expected = expected_stream_bytes(size, bits)
        if len(stream) != expected:
            raise ValueError(f"stream has {len(stream)} bytes; expected {expected}")
        values = _unpack(payload, 3 * size, bits).reshape(size, 3)
        payload_bits = 3 * size * bits
    else:
        values, payload_bits = _unpack_entropy(payload, size, bits)
    levels = (1 << bits) - 1
    coordinates = values.astype(np.float64) * (2.0 / levels) - 1.0
    return ConstellationPacket(
        coordinates=coordinates,
        bits=bits,
        mode=mode,
        output_points=output_points,
        payload_bits=payload_bits,
        header_bytes=HEADER.size,
        payload_bytes=len(payload),
        stream_bytes=len(stream),
    )
