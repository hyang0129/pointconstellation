from __future__ import annotations

import numpy as np
import pytest

from pointconstellation.range_coder import RangeDecoder, RangeEncoder


def test_range_coder_known_answer() -> None:
    cumulative = (0, 1, 3, 4)
    symbols = (0, 1, 1, 2, 0, 2)
    encoder = RangeEncoder()
    for symbol in symbols:
        encoder.encode(cumulative[symbol], cumulative[symbol + 1], cumulative[-1])

    encoded = encoder.finish()

    assert encoded.hex() == "24d0"
    decoder = RangeDecoder(encoded)
    decoded = []
    for _ in symbols:
        value = decoder.cumulative(cumulative[-1])
        symbol = int(np.searchsorted(cumulative, value, side="right") - 1)
        decoded.append(symbol)
        decoder.decode(cumulative[symbol], cumulative[symbol + 1], cumulative[-1])
    assert decoded == list(symbols)


def test_range_coder_validates_frequency_intervals() -> None:
    encoder = RangeEncoder()
    with pytest.raises(ValueError, match="frequency interval"):
        encoder.encode(2, 2, 3)
    with pytest.raises(ValueError, match="too large"):
        encoder.encode(0, 1, (1 << 30) + 1)
    with pytest.raises(ValueError, match="truncated"):
        RangeDecoder(b"")


def test_range_coder_round_trips_random_frequency_tables() -> None:
    rng = np.random.default_rng(20260826)
    for _ in range(20):
        frequencies = rng.integers(1, 100, size=12)
        cumulative = np.concatenate(([0], np.cumsum(frequencies)))
        symbols = rng.integers(0, len(frequencies), size=80)
        encoder = RangeEncoder()
        for symbol in symbols:
            encoder.encode(
                int(cumulative[symbol]),
                int(cumulative[symbol + 1]),
                int(cumulative[-1]),
            )
        decoder = RangeDecoder(encoder.finish())
        decoded = []
        for _ in symbols:
            value = decoder.cumulative(int(cumulative[-1]))
            symbol = int(np.searchsorted(cumulative, value, side="right") - 1)
            decoded.append(symbol)
            decoder.decode(
                int(cumulative[symbol]),
                int(cumulative[symbol + 1]),
                int(cumulative[-1]),
            )
        assert decoded == symbols.tolist()
