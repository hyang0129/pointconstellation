"""Deterministic fixed-width bitstream for an ordered learned feature latent."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAGIC = b"PFLT"
VERSION = 1
HEADER = struct.Struct(">4sBBHI")


@dataclass(frozen=True)
class FeaturePacket:
    """Decoded feature vector and all serialized metadata."""

    features: NDArray[np.float64]
    bits: int
    output_points: int
    payload_bits: int
    header_bytes: int
    payload_bytes: int
    stream_bytes: int


def expected_feature_payload_bytes(latent_dim: int, bits: int) -> int:
    """Return the byte-aligned feature payload size without the header."""

    if not 1 <= latent_dim <= 65535:
        raise ValueError("latent_dim must be between 1 and 65535")
    if not 2 <= bits <= 16:
        raise ValueError("bits must be between 2 and 16")
    return math.ceil(latent_dim * bits / 8)


def expected_feature_stream_bytes(latent_dim: int, bits: int) -> int:
    return HEADER.size + expected_feature_payload_bytes(latent_dim, bits)


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
                raise ValueError("truncated feature payload")
            accumulator = (accumulator << 8) | payload[offset]
            buffered += 8
            offset += 1
        buffered -= bits
        values[index] = (accumulator >> buffered) & ((1 << bits) - 1)
        accumulator &= (1 << buffered) - 1 if buffered else 0
    if buffered and accumulator:
        raise ValueError("non-zero feature padding bits")
    if any(payload[offset:]):
        raise ValueError("unexpected non-zero bytes after feature payload")
    return values


def encode_features(
    features: ArrayLike,
    *,
    bits: int,
    output_points: int,
) -> bytes:
    """Serialize an ordered quantized feature vector in the ``[-1, 1]`` domain."""

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("features must be a nonempty vector")
    if not np.isfinite(array).all():
        raise ValueError("features must be finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("features must lie in the declared [-1, 1] domain")
    expected_feature_stream_bytes(len(array), bits)
    if not 1 <= output_points <= 0xFFFFFFFF:
        raise ValueError("output_points must fit the stream")
    levels = (1 << bits) - 1
    lattice = np.rint((array + 1.0) * 0.5 * levels).astype(np.uint32)
    return HEADER.pack(MAGIC, VERSION, bits, len(array), output_points) + _pack(
        lattice, bits
    )


def decode_features(stream: bytes) -> FeaturePacket:
    """Decode and validate a learned feature-latent stream."""

    if len(stream) < HEADER.size:
        raise ValueError("truncated feature header")
    magic, version, bits, latent_dim, output_points = HEADER.unpack_from(stream)
    if magic != MAGIC:
        raise ValueError("invalid feature magic")
    if version != VERSION:
        raise ValueError(f"unsupported feature version: {version}")
    expected = expected_feature_stream_bytes(latent_dim, bits)
    if len(stream) != expected:
        raise ValueError(f"stream has {len(stream)} bytes; expected {expected}")
    if output_points < 1:
        raise ValueError("feature stream output point count must be positive")
    values = _unpack(stream[HEADER.size :], latent_dim, bits)
    levels = (1 << bits) - 1
    features = values.astype(np.float64) * (2.0 / levels) - 1.0
    return FeaturePacket(
        features=features,
        bits=bits,
        output_points=output_points,
        payload_bits=latent_dim * bits,
        header_bytes=HEADER.size,
        payload_bytes=len(stream) - HEADER.size,
        stream_bytes=len(stream),
    )
