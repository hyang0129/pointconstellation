from __future__ import annotations

import numpy as np
import pytest

from pointconstellation.bitstream import (
    HEADER,
    MODE_ENTROPY,
    MODE_FIXED,
    decode_constellation,
    encode_constellation,
    entropy_bound_bytes,
    expected_stream_bytes,
)


def _lattice_coordinates(lattice: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    return lattice.astype(np.float64) * (2.0 / levels) - 1.0


def _decoded_lattice(stream: bytes) -> np.ndarray:
    packet = decode_constellation(stream)
    levels = (1 << packet.bits) - 1
    return np.rint((packet.coordinates + 1.0) * 0.5 * levels).astype(np.uint32)


@pytest.mark.parametrize("bits", [2, 7, 12, 24])
def test_fixed_width_stream_round_trips_lattice(bits: int) -> None:
    points = np.asarray(
        [[-1.0, 0.0, 1.0], [0.3, -0.6, 0.9], [-0.2, 0.1, -0.8]],
        dtype=np.float32,
    )
    stream = encode_constellation(points, bits=bits, mode=MODE_FIXED, output_points=32)
    packet = decode_constellation(stream)
    levels = (1 << bits) - 1
    expected = np.rint((points.astype(np.float64) + 1.0) * 0.5 * levels).astype(
        np.uint32
    )

    assert packet.mode == MODE_FIXED
    assert packet.bits == bits
    assert packet.output_points == 32
    assert packet.payload_bits == 9 * bits
    assert packet.header_bytes == HEADER.size
    assert packet.header_bytes + packet.payload_bytes == expected_stream_bytes(3, bits)
    assert len(stream) == expected_stream_bytes(3, bits)
    assert sorted(map(tuple, _decoded_lattice(stream).tolist())) == sorted(
        map(tuple, expected.tolist())
    )


@pytest.mark.parametrize("bits", [8, 10, 12])
@pytest.mark.parametrize("constellation_size", [4, 8, 16, 32])
def test_entropy_stream_round_trips_random_lattices(
    bits: int, constellation_size: int
) -> None:
    rng = np.random.default_rng(10_000 * bits + constellation_size)
    levels = (1 << bits) - 1
    lattice = rng.integers(0, levels + 1, size=(constellation_size, 3), dtype=np.uint32)
    lattice[0] = 0
    lattice[1] = levels
    lattice[2] = lattice[0]
    coordinates = _lattice_coordinates(lattice, bits)

    stream = encode_constellation(
        coordinates,
        bits=bits,
        mode=MODE_ENTROPY,
        output_points=2048,
    )
    packet = decode_constellation(stream)

    assert packet.mode == MODE_ENTROPY
    assert packet.bits == bits
    assert packet.output_points == 2048
    assert sorted(map(tuple, _decoded_lattice(stream).tolist())) == sorted(
        map(tuple, lattice.tolist())
    )
    assert (
        encode_constellation(
            packet.coordinates,
            bits=packet.bits,
            mode=packet.mode,
            output_points=packet.output_points,
        )
        == stream
    )


@pytest.mark.parametrize("mode", [MODE_FIXED, MODE_ENTROPY])
def test_unordered_constellations_have_one_canonical_stream(mode: int) -> None:
    points = np.asarray(
        [[0.5, -0.25, 0.0], [-0.5, 0.75, 1.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )

    first = encode_constellation(points, bits=10, mode=mode, output_points=2048)
    second = encode_constellation(
        points[[2, 0, 1]], bits=10, mode=mode, output_points=2048
    )

    assert first == second
    assert decode_constellation(first).mode == mode


@pytest.mark.parametrize("mode", ["free", "strict_subset", "fps"])
def test_fixed_stream_preserves_representation_mode(mode: str) -> None:
    points = np.asarray(
        [[0.5, -0.25, 0.0], [-0.5, 0.75, 1.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )

    stream = encode_constellation(points, bits=10, mode=mode, output_points=2048)
    packet = decode_constellation(stream)

    assert packet.mode == mode
    assert len(stream) == expected_stream_bytes(3, 10)
    assert (
        encode_constellation(
            packet.coordinates,
            bits=packet.bits,
            mode=packet.mode,
            output_points=packet.output_points,
        )
        == stream
    )


def test_entropy_stream_saves_bytes_on_clustered_points() -> None:
    bits = 12
    center = 1 << (bits - 1)
    offsets = np.asarray(
        [[x, y, z] for x in range(4) for y in range(4) for z in range(2)],
        dtype=np.uint32,
    )
    coordinates = _lattice_coordinates(center + offsets, bits)

    fixed = encode_constellation(
        coordinates, bits=bits, mode=MODE_FIXED, output_points=2048
    )
    entropy = encode_constellation(
        coordinates, bits=bits, mode=MODE_ENTROPY, output_points=2048
    )

    assert len(entropy) <= len(fixed)
    assert len(entropy) >= HEADER.size + 1
    assert entropy_bound_bytes(coordinates, bits=bits) <= len(entropy)


def test_stream_rejects_bad_lengths_and_padding() -> None:
    stream = encode_constellation(
        np.zeros((1, 3), dtype=np.float32),
        bits=3,
        mode=MODE_FIXED,
        output_points=8,
    )
    bad_padding = stream[:-1] + bytes([stream[-1] | 1])

    with pytest.raises(ValueError, match="padding"):
        decode_constellation(bad_padding)
    with pytest.raises(ValueError, match="expected"):
        decode_constellation(stream + b"\0")
    with pytest.raises(ValueError, match="header"):
        decode_constellation(stream[: HEADER.size - 1])


def test_stream_validates_domain_mode_and_output_size() -> None:
    with pytest.raises(ValueError, match="unknown constellation stream mode"):
        encode_constellation([[0.0, 0.0, 0.0]], bits=8, mode=7, output_points=8)
    with pytest.raises(ValueError, match="unknown constellation stream mode"):
        encode_constellation(
            [[0.0, 0.0, 0.0]], bits=8, mode="latent", output_points=8
        )
    with pytest.raises(ValueError, match="domain"):
        encode_constellation(
            [[2.0, 0.0, 0.0]], bits=8, mode=MODE_FIXED, output_points=8
        )
    with pytest.raises(ValueError, match="at least K"):
        encode_constellation(np.zeros((4, 3)), bits=8, mode=MODE_FIXED, output_points=2)


def test_decoder_rejects_unknown_stream_mode() -> None:
    stream = encode_constellation(
        [[0.0, 0.0, 0.0]], bits=8, mode=MODE_FIXED, output_points=8
    )
    unknown_mode = stream[:5] + bytes([255]) + stream[6:]

    with pytest.raises(ValueError, match="unknown constellation stream mode"):
        decode_constellation(unknown_mode)
