from __future__ import annotations

import numpy as np
import pytest

from pointconstellation.bitstream import (
    HEADER,
    decode_constellation,
    encode_constellation,
    expected_stream_bytes,
)


@pytest.mark.parametrize("bits", [2, 7, 12, 24])
def test_fixed_width_stream_round_trips_lattice(bits: int) -> None:
    points = np.asarray(
        [[-1.0, 0.0, 1.0], [0.3, -0.6, 0.9], [-0.2, 0.1, -0.8]],
        dtype=np.float32,
    )
    stream = encode_constellation(points, bits=bits, mode="free", output_points=32)
    packet = decode_constellation(stream)
    levels = (1 << bits) - 1
    expected = np.rint((points.astype(np.float64) + 1.0) * 0.5 * levels).astype(
        np.uint32
    )
    actual = np.rint((packet.coordinates + 1.0) * 0.5 * levels).astype(np.uint32)

    assert packet.mode == "free"
    assert packet.bits == bits
    assert packet.output_points == 32
    assert packet.payload_bits == 9 * bits
    assert packet.header_bytes == HEADER.size
    assert packet.header_bytes + packet.payload_bytes == expected_stream_bytes(3, bits)
    assert len(stream) == expected_stream_bytes(3, bits)
    assert sorted(map(tuple, actual.tolist())) == sorted(map(tuple, expected.tolist()))


def test_unordered_constellations_have_one_canonical_stream() -> None:
    points = np.asarray(
        [[0.5, -0.25, 0.0], [-0.5, 0.75, 1.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )

    first = encode_constellation(
        points, bits=10, mode="strict_subset", output_points=2048
    )
    second = encode_constellation(
        points[[2, 0, 1]], bits=10, mode="strict_subset", output_points=2048
    )

    assert first == second
    assert decode_constellation(first).mode == "strict_subset"


def test_stream_rejects_bad_lengths_and_padding() -> None:
    stream = encode_constellation(
        np.zeros((1, 3), dtype=np.float32),
        bits=3,
        mode="fps",
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
    with pytest.raises(ValueError, match="unknown constellation mode"):
        encode_constellation([[0.0, 0.0, 0.0]], bits=8, mode="latent", output_points=8)
    with pytest.raises(ValueError, match="domain"):
        encode_constellation([[2.0, 0.0, 0.0]], bits=8, mode="fps", output_points=8)
    with pytest.raises(ValueError, match="at least K"):
        encode_constellation(np.zeros((4, 3)), bits=8, mode="fps", output_points=2)
