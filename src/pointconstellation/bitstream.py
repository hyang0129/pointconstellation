"""Deterministic fixed-width and entropy-coded constellation streams.

All stream modes quantize coordinates on the declared ``[-1, 1]`` lattice
and sort them lexicographically.  Mode 0 is the declared fixed-width stream.
Mode 1 is an optional diagnostic that delta-codes the same unordered lattice
points with a stream-adaptive Rice code.  The learned mode uses shared integer
probability-model state and an exact arithmetic coder.  No mode carries learned
features or target-only information.

Streams may append the mesh normalization needed to return to the source frame.
The center and isotropic scale are stored as four IEEE 754 binary16 values.  A
decoder therefore restores coordinates with the *serialized*, rounded transform,
not the encoder's full-precision transform.  Original-frame distortion includes
that rounding loss.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.bitstream_bits import BitReader as _BitReader
from pointconstellation.bitstream_bits import BitWriter as _BitWriter
from pointconstellation.learned_entropy import (
    LearnedEntropyModel,
    decode_learned_lattice,
    default_octree_model,
    encode_learned_lattice,
)

MAGIC = b"PCON"
VERSION = 1
MODE_FIXED = 0
MODE_ENTROPY = 1
MODE_LEARNED = 2
MODE_SELECTIVE = 3
MODE_SELECTIVE_ENTROPY = 4
REPRESENTATION_MODES = ("free", "strict_subset", "fps")
MODE_IDS = {
    MODE_FIXED: 0,
    MODE_ENTROPY: 1,
    # Wire IDs 2--4 predate learned coding and identify representation modes.
    # Keep MODE_LEARNED's public value while allocating a collision-free ID.
    MODE_LEARNED: 5,
    MODE_SELECTIVE: 6,
    MODE_SELECTIVE_ENTROPY: 7,
    "free": 2,
    "strict_subset": 3,
    "fps": 4,
}
ID_MODES = {value: key for key, value in MODE_IDS.items()}
CONSTELLATION_MODES = (
    MODE_FIXED,
    MODE_ENTROPY,
    MODE_LEARNED,
    *REPRESENTATION_MODES,
)
DOMAIN_UNIT_CUBE = 1
DOMAIN_UNIT_CUBE_WITH_NORMALIZATION = 2
HEADER = struct.Struct(">4sBBBBHI")
SELECTIVE_HEADER = struct.Struct(">4sBBBBHHI")
SELECTIVE_ENTROPY_LENGTH = struct.Struct(">I")
NORMALIZATION = struct.Struct(">4e")


@dataclass(frozen=True)
class ConstellationPacket:
    """Decoded coordinate payload and the metadata required to interpret it."""

    coordinates: NDArray[np.float64]
    normalized_coordinates: NDArray[np.float64]
    bits: int
    mode: int | str
    output_points: int
    payload_bits: int
    header_bytes: int
    payload_bytes: int
    normalization_bytes: int
    stream_bytes: int
    normalization_center: NDArray[np.float64] | None
    normalization_scale: float | None


@dataclass(frozen=True)
class SelectivePacket:
    """Decoded constellation and raw pass-through coordinate groups."""

    constellation_coordinates: NDArray[np.float64]
    preserved_coordinates: NDArray[np.float64]
    bits: int
    mode: int
    output_points: int
    payload_bits: int
    header_bytes: int
    payload_bytes: int
    stream_bytes: int

    @property
    def k1(self) -> int:
        """Return the number of decoder-input constellation points."""

        return len(self.constellation_coordinates)

    @property
    def k2(self) -> int:
        """Return the number of raw pass-through points."""

        return len(self.preserved_coordinates)

    @property
    def coordinates(self) -> NDArray[np.float64]:
        """Return the two coordinate groups in their count-declared order."""

        return np.concatenate(
            (self.constellation_coordinates, self.preserved_coordinates), axis=0
        )


def expected_payload_bytes(constellation_size: int, bits: int) -> int:
    """Return the byte-aligned fixed-width payload size without the header."""

    if not 1 <= constellation_size <= 65535:
        raise ValueError("constellation_size must be between 1 and 65535")
    if not 2 <= bits <= 24:
        raise ValueError("bits must be between 2 and 24")
    return math.ceil(3 * constellation_size * bits / 8)


def expected_selective_payload_bytes(k1: int, k2: int, bits: int) -> int:
    """Return the fixed-width payload bytes for both selective coordinate sets."""

    if not 0 <= k1 <= 65535 or not 0 <= k2 <= 65535:
        raise ValueError("selective counts must be between 0 and 65535")
    if not 2 <= bits <= 24:
        raise ValueError("bits must be between 2 and 24")
    return math.ceil(3 * (k1 + k2) * bits / 8)


def expected_selective_stream_bytes(k1: int, k2: int, bits: int) -> int:
    """Return the exact selective fixed stream size, including count metadata."""

    payload_bytes = expected_selective_payload_bytes(k1, k2, bits)
    if k1 > 0 and k2 == 0:
        return HEADER.size + payload_bytes
    return SELECTIVE_HEADER.size + payload_bytes


def expected_stream_bytes(
    constellation_size: int, bits: int, *, normalization: bool = False
) -> int:
    """Return the exact fixed-width stream size, including the header."""

    return (
        HEADER.size
        + expected_payload_bytes(constellation_size, bits)
        + (NORMALIZATION.size if normalization else 0)
    )


def _coordinates(points: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("coordinates must have shape (K, 3) with K > 0")
    if not np.isfinite(array).all():
        raise ValueError("coordinates must be finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("coordinates must lie in the declared [-1, 1] domain")
    return array


def _optional_coordinates(points: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (K, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must lie in the declared [-1, 1] domain")
    return array


def _lattice(coordinates: ArrayLike, bits: int) -> NDArray[np.uint32]:
    points = _coordinates(coordinates)
    expected_stream_bytes(len(points), bits)
    levels = (1 << bits) - 1
    lattice = np.rint((points + 1.0) * 0.5 * levels).astype(np.uint32)
    order = np.lexsort((lattice[:, 2], lattice[:, 1], lattice[:, 0]))
    return lattice[order]


def _optional_lattice(
    coordinates: ArrayLike, bits: int, *, name: str
) -> NDArray[np.uint32]:
    points = _optional_coordinates(coordinates, name=name)
    if not 2 <= bits <= 24:
        raise ValueError("bits must be between 2 and 24")
    levels = (1 << bits) - 1
    lattice = np.rint((points + 1.0) * 0.5 * levels).astype(np.uint32)
    order = np.lexsort((lattice[:, 2], lattice[:, 1], lattice[:, 0]))
    return lattice[order]


def encode_normalization(center: ArrayLike, scale: float) -> bytes:
    """Serialize an isotropic original-frame transform as four binary16 values."""

    center_array = np.asarray(center, dtype=np.float64)
    if center_array.shape != (3,) or not np.isfinite(center_array).all():
        raise ValueError("normalization center must contain three finite values")
    scale_value = float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("normalization scale must be finite and positive")
    binary16_limit = np.finfo(np.float16).max
    if np.any(np.abs(center_array) > binary16_limit) or scale_value > binary16_limit:
        raise ValueError("normalization values must be representable as float16")
    payload = NORMALIZATION.pack(*center_array, scale_value)
    if NORMALIZATION.unpack(payload)[3] <= 0:
        raise ValueError("normalization scale rounds to zero in float16")
    return payload


def decode_normalization(payload: bytes) -> tuple[NDArray[np.float64], float]:
    """Decode and validate one binary16 normalization payload."""

    if len(payload) != NORMALIZATION.size:
        raise ValueError(
            f"normalization has {len(payload)} bytes; expected {NORMALIZATION.size}"
        )
    transform = NORMALIZATION.unpack(payload)
    center = np.asarray(transform[:3], dtype=np.float64)
    scale = float(transform[3])
    if not np.isfinite(center).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid normalization payload")
    return center, scale


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
    mode: int | str = MODE_FIXED,
    output_points: int,
    normalization_center: ArrayLike | None = None,
    normalization_scale: float | None = None,
    learned_model: LearnedEntropyModel | None = None,
) -> bytes:
    """Encode an unordered normalized coordinate set into a canonical stream.

    ``normalization_center`` and ``normalization_scale`` must be supplied
    together.  They declare ``original = normalized * scale + center`` and are
    rounded to binary16 in the stream.
    """

    lattice = _lattice(coordinates, bits)
    constellation_size = len(lattice)
    has_normalization = (normalization_center is not None) or (
        normalization_scale is not None
    )
    if (normalization_center is None) != (normalization_scale is None):
        raise ValueError("normalization center and scale must be supplied together")
    normalization_payload = b""
    if has_normalization:
        normalization_payload = encode_normalization(
            normalization_center, float(normalization_scale)
        )
    expected_stream_bytes(constellation_size, bits, normalization=has_normalization)
    if mode not in CONSTELLATION_MODES:
        raise ValueError(f"unknown constellation stream mode: {mode}")
    if not constellation_size <= output_points <= 0xFFFFFFFF:
        raise ValueError("output_points must fit the stream and be at least K")

    header = HEADER.pack(
        MAGIC,
        VERSION,
        MODE_IDS[mode],
        bits,
        (
            DOMAIN_UNIT_CUBE_WITH_NORMALIZATION
            if has_normalization
            else DOMAIN_UNIT_CUBE
        ),
        constellation_size,
        output_points,
    )
    if mode == MODE_ENTROPY:
        payload, _ = _pack_entropy(lattice, bits)
    elif mode == MODE_LEARNED:
        model = learned_model or default_octree_model(bits, constellation_size)
        model.validate_stream(bits, constellation_size)
        minimum_payload_bytes = max(
            0, math.ceil(entropy_bound_bytes(coordinates, bits=bits)) - HEADER.size
        )
        payload = encode_learned_lattice(
            lattice,
            model=model,
            header=header,
            minimum_payload_bytes=minimum_payload_bytes,
        )
    else:
        payload = _pack(lattice.reshape(-1), bits)
    return header + payload + normalization_payload


def encode_selective(
    constellation_coordinates: ArrayLike,
    preserved_coordinates: ArrayLike,
    *,
    bits: int,
    output_points: int,
    entropy: bool = False,
) -> bytes:
    """Encode a count-delimited constellation plus raw pass-through set.

    Each group is independently lexicographically sorted.  Counts, rather than
    a per-point type channel, delimit the two groups.  A zero-length preserved
    group reduces exactly to the existing mode-0 or mode-1 stream.
    """

    constellation = _optional_lattice(
        constellation_coordinates, bits, name="constellation_coordinates"
    )
    preserved = _optional_lattice(
        preserved_coordinates, bits, name="preserved_coordinates"
    )
    k1 = len(constellation)
    k2 = len(preserved)
    expected_selective_payload_bytes(k1, k2, bits)
    if not 0 <= output_points <= 0xFFFFFFFF:
        raise ValueError("output_points must fit the selective stream")
    if k1 and output_points < k1:
        raise ValueError("output_points must be at least K1")
    levels = (1 << bits) - 1
    if k1 > 0 and k2 == 0:
        coordinates = constellation.astype(np.float64) * (2.0 / levels) - 1.0
        return encode_constellation(
            coordinates,
            bits=bits,
            mode=MODE_ENTROPY if entropy else MODE_FIXED,
            output_points=output_points,
        )

    mode = MODE_SELECTIVE_ENTROPY if entropy else MODE_SELECTIVE
    header = SELECTIVE_HEADER.pack(
        MAGIC,
        VERSION,
        MODE_IDS[mode],
        bits,
        DOMAIN_UNIT_CUBE,
        k1,
        k2,
        output_points,
    )
    if entropy:
        first_payload = _pack_entropy(constellation, bits)[0] if k1 else b""
        second_payload = _pack_entropy(preserved, bits)[0] if k2 else b""
        payload = (
            SELECTIVE_ENTROPY_LENGTH.pack(len(first_payload))
            + first_payload
            + second_payload
        )
    else:
        values = np.concatenate((constellation, preserved), axis=0).reshape(-1)
        payload = _pack(values, bits)
    return header + payload


def _selective_packet(
    *,
    constellation: NDArray[np.uint32],
    preserved: NDArray[np.uint32],
    bits: int,
    mode: int,
    output_points: int,
    payload_bits: int,
    header_bytes: int,
    payload_bytes: int,
    stream_bytes: int,
) -> SelectivePacket:
    levels = (1 << bits) - 1
    scale = 2.0 / levels
    return SelectivePacket(
        constellation_coordinates=constellation.astype(np.float64) * scale - 1.0,
        preserved_coordinates=preserved.astype(np.float64) * scale - 1.0,
        bits=bits,
        mode=mode,
        output_points=output_points,
        payload_bits=payload_bits,
        header_bytes=header_bytes,
        payload_bytes=payload_bytes,
        stream_bytes=stream_bytes,
    )


def _decode_selective_stream(stream: bytes) -> SelectivePacket:
    if len(stream) < SELECTIVE_HEADER.size:
        raise ValueError("truncated selective header")
    magic, version, mode_id, bits, domain, k1, k2, output_points = (
        SELECTIVE_HEADER.unpack_from(stream)
    )
    if magic != MAGIC:
        raise ValueError("invalid constellation magic")
    if version != VERSION:
        raise ValueError(f"unsupported constellation version: {version}")
    mode = ID_MODES.get(mode_id)
    if mode not in {MODE_SELECTIVE, MODE_SELECTIVE_ENTROPY}:
        raise ValueError(f"unknown selective stream mode: {mode_id}")
    if domain != DOMAIN_UNIT_CUBE:
        raise ValueError(f"unsupported selective coordinate domain id: {domain}")
    expected_selective_payload_bytes(k1, k2, bits)
    if k1 and output_points < k1:
        raise ValueError("stream output point count is smaller than K1")
    payload = stream[SELECTIVE_HEADER.size :]
    if mode == MODE_SELECTIVE:
        expected = SELECTIVE_HEADER.size + expected_selective_payload_bytes(
            k1, k2, bits
        )
        if len(stream) != expected:
            raise ValueError(f"stream has {len(stream)} bytes; expected {expected}")
        values = _unpack(payload, 3 * (k1 + k2), bits).reshape(k1 + k2, 3)
        constellation = values[:k1]
        preserved = values[k1:]
        payload_bits = 3 * (k1 + k2) * bits
    else:
        if len(payload) < SELECTIVE_ENTROPY_LENGTH.size:
            raise ValueError("truncated selective entropy block length")
        (first_bytes,) = SELECTIVE_ENTROPY_LENGTH.unpack_from(payload)
        blocks = payload[SELECTIVE_ENTROPY_LENGTH.size :]
        if first_bytes > len(blocks):
            raise ValueError("selective entropy block length exceeds payload")
        first = blocks[:first_bytes]
        second = blocks[first_bytes:]
        if bool(first) != bool(k1) or bool(second) != bool(k2):
            raise ValueError("selective entropy blocks disagree with K1/K2")
        if k1:
            constellation, first_bits = _unpack_entropy(first, k1, bits)
        else:
            constellation = np.empty((0, 3), dtype=np.uint32)
            first_bits = 0
        if k2:
            preserved, second_bits = _unpack_entropy(second, k2, bits)
        else:
            preserved = np.empty((0, 3), dtype=np.uint32)
            second_bits = 0
        payload_bits = 8 * SELECTIVE_ENTROPY_LENGTH.size + first_bits + second_bits
    packet = _selective_packet(
        constellation=constellation,
        preserved=preserved,
        bits=bits,
        mode=mode,
        output_points=output_points,
        payload_bits=payload_bits,
        header_bytes=SELECTIVE_HEADER.size,
        payload_bytes=len(payload),
        stream_bytes=len(stream),
    )
    canonical = encode_selective(
        packet.constellation_coordinates,
        packet.preserved_coordinates,
        bits=bits,
        output_points=output_points,
        entropy=mode == MODE_SELECTIVE_ENTROPY,
    )
    if canonical != stream:
        raise ValueError("non-canonical or corrupt selective payload")
    return packet


def decode_selective(stream: bytes) -> SelectivePacket:
    """Decode a selective stream, including its K2=0 mode-0/1 reduction."""

    packet = decode_constellation(stream)
    if isinstance(packet, SelectivePacket):
        return packet
    if packet.mode not in {MODE_FIXED, MODE_ENTROPY}:
        raise ValueError("a reduced selective stream must use mode 0 or mode 1")
    empty = np.empty((0, 3), dtype=np.float64)
    return SelectivePacket(
        constellation_coordinates=packet.normalized_coordinates,
        preserved_coordinates=empty,
        bits=packet.bits,
        mode=int(packet.mode),
        output_points=packet.output_points,
        payload_bits=packet.payload_bits,
        header_bytes=packet.header_bytes,
        payload_bytes=packet.payload_bytes,
        stream_bytes=packet.stream_bytes,
    )


def decode_constellation(
    stream: bytes, *, learned_model: LearnedEntropyModel | None = None
) -> ConstellationPacket | SelectivePacket:
    """Decode and validate any supported constellation stream."""

    if len(stream) < HEADER.size:
        raise ValueError("truncated constellation header")
    mode_id = stream[5]
    if ID_MODES.get(mode_id) in {MODE_SELECTIVE, MODE_SELECTIVE_ENTROPY}:
        return _decode_selective_stream(stream)
    magic, version, mode_id, bits, domain, size, output_points = HEADER.unpack_from(
        stream
    )
    if magic != MAGIC:
        raise ValueError("invalid constellation magic")
    if version != VERSION:
        raise ValueError(f"unsupported constellation version: {version}")
    if mode_id not in ID_MODES:
        raise ValueError(f"unknown constellation stream mode: {mode_id}")
    mode = ID_MODES[mode_id]
    if domain not in {DOMAIN_UNIT_CUBE, DOMAIN_UNIT_CUBE_WITH_NORMALIZATION}:
        raise ValueError(f"unsupported coordinate domain id: {domain}")
    has_normalization = domain == DOMAIN_UNIT_CUBE_WITH_NORMALIZATION
    expected_payload_bytes(size, bits)
    if output_points < size:
        raise ValueError("stream output point count is smaller than K")

    normalization_bytes = NORMALIZATION.size if has_normalization else 0
    payload_end = len(stream) - normalization_bytes
    if payload_end < HEADER.size:
        raise ValueError("truncated constellation payload")
    payload = stream[HEADER.size : payload_end]
    if mode == MODE_ENTROPY:
        values, payload_bits = _unpack_entropy(payload, size, bits)
    elif mode == MODE_LEARNED:
        model = learned_model or default_octree_model(bits, size)
        model.validate_stream(bits, size)
        values = decode_learned_lattice(
            payload,
            model=model,
            header=stream[: HEADER.size],
        )
        payload_bits = 8 * len(payload)
    else:
        expected = expected_stream_bytes(size, bits, normalization=has_normalization)
        if len(stream) != expected:
            raise ValueError(f"stream has {len(stream)} bytes; expected {expected}")
        values = _unpack(payload, 3 * size, bits).reshape(size, 3)
        payload_bits = 3 * size * bits
    levels = (1 << bits) - 1
    normalized = values.astype(np.float64) * (2.0 / levels) - 1.0
    center = None
    scale = None
    coordinates = normalized
    if has_normalization:
        center, scale = decode_normalization(stream[payload_end:])
        coordinates = normalized * scale + center
    if mode == MODE_LEARNED:
        canonical = encode_constellation(
            normalized,
            bits=bits,
            mode=mode,
            output_points=output_points,
            normalization_center=center,
            normalization_scale=scale,
            learned_model=model,
        )
        if canonical != stream:
            raise ValueError("non-canonical or corrupt learned constellation payload")
    return ConstellationPacket(
        coordinates=coordinates,
        normalized_coordinates=normalized,
        bits=bits,
        mode=mode,
        output_points=output_points,
        payload_bits=payload_bits,
        header_bytes=HEADER.size,
        payload_bytes=len(payload),
        normalization_bytes=normalization_bytes,
        stream_bytes=len(stream),
        normalization_center=center,
        normalization_scale=scale,
    )
