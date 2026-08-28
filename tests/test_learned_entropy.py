from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pointconstellation.bitstream import (
    HEADER,
    MODE_FIXED,
    MODE_LEARNED,
    NORMALIZATION,
    decode_constellation,
    encode_constellation,
    entropy_bound_bytes,
)
from pointconstellation.learned_entropy import (
    AUTOREGRESSIVE,
    OCTREE,
    LearnedEntropyConfig,
    LearnedEntropyModel,
    fit_learned_entropy_model,
)

FIXTURE = (
    Path(__file__).parent / "fixtures/experiment_019_validation_constellation.json"
)


def _coordinates(lattice: np.ndarray, bits: int) -> np.ndarray:
    return lattice.astype(np.float64) * (2.0 / ((1 << bits) - 1)) - 1.0


def _decoded_lattice(
    stream: bytes, model: LearnedEntropyModel | None = None
) -> np.ndarray:
    packet = decode_constellation(stream, learned_model=model)
    levels = (1 << packet.bits) - 1
    return np.rint((packet.coordinates + 1.0) * 0.5 * levels).astype(np.uint32)


@pytest.mark.parametrize("bits", [8, 10, 12])
@pytest.mark.parametrize("constellation_size", [4, 8, 16, 32])
def test_mode_2_round_trips_random_lattices_with_duplicates(
    bits: int, constellation_size: int
) -> None:
    rng = np.random.default_rng(1000 * bits + constellation_size)
    levels = (1 << bits) - 1
    lattice = rng.integers(0, levels + 1, size=(constellation_size, 3), dtype=np.uint32)
    lattice[1] = lattice[0]
    stream = encode_constellation(
        _coordinates(lattice, bits),
        bits=bits,
        mode=MODE_LEARNED,
        output_points=2048,
    )

    packet = decode_constellation(stream)

    assert packet.mode == MODE_LEARNED
    assert packet.header_bytes == HEADER.size
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


@pytest.mark.parametrize("candidate", [OCTREE, AUTOREGRESSIVE])
def test_fitted_candidates_are_exact_deterministic_and_persistent(
    candidate: str, tmp_path: Path
) -> None:
    rng = np.random.default_rng(51)
    bits = 8
    training = rng.integers(96, 144, size=(48, 4, 3), dtype=np.uint32)
    model = fit_learned_entropy_model(
        training,
        bits=bits,
        config=LearnedEntropyConfig(candidate=candidate, hidden_width=8),
    )
    coordinates = _coordinates(training[0], bits)

    first = encode_constellation(
        coordinates,
        bits=bits,
        mode=MODE_LEARNED,
        output_points=32,
        learned_model=model,
    )
    second = encode_constellation(
        coordinates[[3, 1, 0, 2]],
        bits=bits,
        mode=MODE_LEARNED,
        output_points=32,
        learned_model=model,
    )
    path = tmp_path / f"{candidate}.npz"
    model.save(path)
    loaded = LearnedEntropyModel.load(path)

    assert first == second
    assert loaded.model_hash == model.model_hash
    assert loaded.parameter_bytes == model.parameter_bytes
    assert (
        encode_constellation(
            coordinates,
            bits=bits,
            mode=MODE_LEARNED,
            output_points=32,
            learned_model=loaded,
        )
        == first
    )
    assert sorted(map(tuple, _decoded_lattice(first, loaded).tolist())) == sorted(
        map(tuple, training[0].tolist())
    )


def test_mode_2_rejects_corruption_and_wrong_shared_model() -> None:
    rng = np.random.default_rng(82)
    training = rng.integers(40, 180, size=(32, 4, 3), dtype=np.uint32)
    model = fit_learned_entropy_model(
        training,
        bits=8,
        config=LearnedEntropyConfig(candidate=OCTREE),
    )
    stream = encode_constellation(
        _coordinates(training[0], 8),
        bits=8,
        mode=MODE_LEARNED,
        output_points=32,
        learned_model=model,
    )
    corrupted = bytearray(stream)
    corrupted[-1] ^= 1

    with pytest.raises(ValueError, match="checksum"):
        decode_constellation(bytes(corrupted), learned_model=model)
    with pytest.raises(ValueError, match="truncated|checksum|corrupt"):
        decode_constellation(stream[:-1], learned_model=model)
    with pytest.raises(ValueError, match="checksum|corrupt|canonical"):
        decode_constellation(stream)


def test_learned_mode_has_distinct_wire_id_from_representation_modes() -> None:
    coordinates = np.zeros((2, 3), dtype=np.float64)
    learned = encode_constellation(
        coordinates,
        bits=8,
        mode=MODE_LEARNED,
        output_points=32,
    )
    free = encode_constellation(
        coordinates,
        bits=8,
        mode="free",
        output_points=32,
    )

    assert HEADER.unpack_from(learned)[2] != HEADER.unpack_from(free)[2]
    assert decode_constellation(learned).mode == MODE_LEARNED
    assert decode_constellation(free).mode == "free"


def test_learned_stream_preserves_serialized_normalization() -> None:
    bits = 10
    coordinates = np.asarray(
        [[-0.5, 0.0, 0.25], [0.25, 0.5, 0.75]], dtype=np.float64
    )
    stream = encode_constellation(
        coordinates,
        bits=bits,
        mode=MODE_LEARNED,
        output_points=32,
        normalization_center=[4.5, -2.0, 0.25],
        normalization_scale=1.75,
    )

    packet = decode_constellation(stream)

    assert packet.mode == MODE_LEARNED
    assert packet.normalization_bytes == NORMALIZATION.size
    assert np.allclose(
        packet.coordinates,
        packet.normalized_coordinates * packet.normalization_scale
        + packet.normalization_center,
    )
    assert (
        encode_constellation(
            packet.normalized_coordinates,
            bits=packet.bits,
            mode=packet.mode,
            output_points=packet.output_points,
            normalization_center=packet.normalization_center,
            normalization_scale=packet.normalization_scale,
        )
        == stream
    )


def test_real_fixture_and_clustered_streams_save_bytes_without_crossing_bound() -> None:
    fixture = json.loads(FIXTURE.read_text())
    bits = fixture["bits"]
    real = np.asarray(fixture["lattice"], dtype=np.uint32)
    center = 1 << (bits - 1)
    clustered = center + np.asarray(
        [[x, y, z] for x in range(4) for y in range(2) for z in range(1)],
        dtype=np.uint32,
    )
    training = np.concatenate(
        (
            np.repeat(real[None, :, :], 32, axis=0),
            np.repeat(clustered[None, :, :], 32, axis=0),
        )
    )
    model = fit_learned_entropy_model(
        training,
        bits=bits,
        config=LearnedEntropyConfig(candidate=AUTOREGRESSIVE, hidden_width=8),
    )

    for lattice in (real, clustered):
        coordinates = _coordinates(lattice, bits)
        fixed = encode_constellation(
            coordinates, bits=bits, mode=MODE_FIXED, output_points=2048
        )
        learned = encode_constellation(
            coordinates,
            bits=bits,
            mode=MODE_LEARNED,
            output_points=2048,
            learned_model=model,
        )
        packet = decode_constellation(learned, learned_model=model)

        assert len(learned) < len(fixed)
        assert len(learned) >= math.ceil(entropy_bound_bytes(coordinates, bits=bits))
        assert np.array_equal(
            packet.coordinates, decode_constellation(fixed).coordinates
        )
