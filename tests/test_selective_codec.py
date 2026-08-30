from __future__ import annotations

import numpy as np

from pointconstellation.bitstream import (
    MODE_ENTROPY,
    MODE_FIXED,
    MODE_SELECTIVE,
    MODE_SELECTIVE_ENTROPY,
    decode_selective,
    encode_constellation,
    encode_selective,
    expected_selective_payload_bytes,
    expected_selective_stream_bytes,
)
from pointconstellation.selective_codec import decode_selective_message


def _coordinates(count: int, bits: int, offset: int) -> np.ndarray:
    levels = (1 << bits) - 1
    lattice = np.empty((count, 3), dtype=np.uint32)
    for index in range(count):
        lattice[index] = (
            (17 * index + offset) % (levels + 1),
            (31 * index + 2 * offset) % (levels + 1),
            (47 * index + 3 * offset) % (levels + 1),
        )
    return lattice.astype(np.float64) * (2.0 / levels) - 1.0


def _lattice(points: np.ndarray, bits: int) -> list[tuple[int, int, int]]:
    levels = (1 << bits) - 1
    values = np.rint((points + 1.0) * 0.5 * levels).astype(np.uint32)
    return sorted(map(tuple, values.tolist()))


def test_selective_fixed_round_trips_all_counts_through_32() -> None:
    bits = 12
    for k1 in range(33):
        for k2 in range(33):
            constellation = _coordinates(k1, bits, 11)
            preserved = _coordinates(k2, bits, 101)
            output_points = max(64, k1)
            stream = encode_selective(
                constellation,
                preserved,
                bits=bits,
                output_points=output_points,
            )
            packet = decode_selective(stream)

            assert packet.k1 == k1
            assert packet.k2 == k2
            assert packet.output_points == output_points
            assert _lattice(packet.constellation_coordinates, bits) == _lattice(
                constellation, bits
            )
            assert _lattice(packet.preserved_coordinates, bits) == _lattice(
                preserved, bits
            )
            assert packet.payload_bytes == expected_selective_payload_bytes(
                k1, k2, bits
            )
            assert len(stream) == expected_selective_stream_bytes(k1, k2, bits)
            assert (
                encode_selective(
                    packet.constellation_coordinates,
                    packet.preserved_coordinates,
                    bits=bits,
                    output_points=output_points,
                )
                == stream
            )
            assert packet.mode == (MODE_FIXED if k1 > 0 and k2 == 0 else MODE_SELECTIVE)


def test_selective_entropy_is_canonical_and_k2_zero_reduces_to_mode_one() -> None:
    constellation = _coordinates(7, 10, 3)
    preserved = _coordinates(5, 10, 71)
    first = encode_selective(
        constellation,
        preserved,
        bits=10,
        output_points=32,
        entropy=True,
    )
    second = encode_selective(
        constellation[::-1],
        preserved[[2, 4, 0, 3, 1]],
        bits=10,
        output_points=32,
        entropy=True,
    )
    packet = decode_selective(first)

    assert first == second
    assert packet.mode == MODE_SELECTIVE_ENTROPY
    assert packet.k1 == 7 and packet.k2 == 5
    assert (
        encode_selective(
            packet.constellation_coordinates,
            packet.preserved_coordinates,
            bits=10,
            output_points=32,
            entropy=True,
        )
        == first
    )

    reduced = encode_selective(
        constellation,
        np.empty((0, 3)),
        bits=10,
        output_points=32,
        entropy=True,
    )
    expected = encode_constellation(
        constellation,
        bits=10,
        mode=MODE_ENTROPY,
        output_points=32,
    )
    assert reduced == expected
    assert decode_selective(reduced).mode == MODE_ENTROPY


def test_selective_decoder_preserves_raw_lattice_points_exactly() -> None:
    constellation = _coordinates(3, 8, 5)
    preserved = _coordinates(4, 8, 31)
    stream = encode_selective(constellation, preserved, bits=8, output_points=6)

    def decoder(points: np.ndarray, output_points: int) -> np.ndarray:
        repeats = int(np.ceil(output_points / len(points)))
        return np.tile(points, (repeats, 1))[:output_points]

    result = decode_selective_message(stream, decoder=decoder)

    assert result.preservation_error == 0.0
    assert np.array_equal(
        result.reconstruction[-result.packet.k2 :],
        result.packet.preserved_coordinates,
    )

    pure_stream = encode_selective(np.empty((0, 3)), preserved, bits=8, output_points=0)
    pure = decode_selective_message(pure_stream, decoder=None)
    assert np.array_equal(pure.reconstruction, pure.packet.preserved_coordinates)
