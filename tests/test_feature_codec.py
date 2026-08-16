from __future__ import annotations

import numpy as np
import pytest
import torch

from pointconstellation.feature_bitstream import (
    HEADER,
    decode_features,
    encode_features,
    expected_feature_stream_bytes,
)
from pointconstellation.models.feature_codec import VariableFeatureCodec


@pytest.mark.parametrize("bits", [2, 7, 8, 16])
def test_feature_stream_round_trips_exact_lattice(bits: int) -> None:
    features = np.asarray([-1.0, -0.3, 0.0, 0.7, 1.0], dtype=np.float32)
    stream = encode_features(features, bits=bits, output_points=2048)
    packet = decode_features(stream)
    levels = (1 << bits) - 1
    expected = np.rint((features.astype(np.float64) + 1.0) * 0.5 * levels)
    actual = np.rint((packet.features + 1.0) * 0.5 * levels)

    assert np.array_equal(actual, expected)
    assert packet.output_points == 2048
    assert packet.payload_bits == len(features) * bits
    assert len(stream) == expected_feature_stream_bytes(len(features), bits)


def test_feature_stream_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="vector"):
        encode_features(np.zeros((2, 3)), bits=8, output_points=8)
    with pytest.raises(ValueError, match="domain"):
        encode_features([2.0], bits=8, output_points=8)
    stream = encode_features([0.0], bits=3, output_points=8)
    with pytest.raises(ValueError, match="padding"):
        decode_features(stream[:-1] + bytes([stream[-1] | 1]))
    with pytest.raises(ValueError, match="header"):
        decode_features(stream[: HEADER.size - 1])


def test_feature_codec_is_permutation_invariant_and_variable_rate() -> None:
    torch.manual_seed(4)
    codec = VariableFeatureCodec(32, 20, bits=8, feature_width=16).eval()
    points = torch.rand(2, 32, 3) * 2.0 - 1.0

    reconstruction, features = codec(points, 8)
    permuted_reconstruction, permuted_features = codec(points[:, torch.randperm(32)], 8)
    larger_reconstruction, larger_features = codec(points, 20)

    assert reconstruction.shape == larger_reconstruction.shape == (2, 32, 3)
    assert features.shape == (2, 8)
    assert larger_features.shape == (2, 20)
    assert torch.equal(features, permuted_features)
    assert torch.equal(reconstruction, permuted_reconstruction)
    assert not torch.equal(reconstruction, larger_reconstruction)


def test_feature_codec_backpropagates_through_quantization() -> None:
    codec = VariableFeatureCodec(16, 12, bits=8, feature_width=16).train()
    points = torch.rand(2, 16, 3) * 2.0 - 1.0
    reconstruction, features = codec(points, 6)
    (reconstruction.square().mean() + features.square().mean()).backward()

    assert all(parameter.grad is not None for parameter in codec.parameters())
