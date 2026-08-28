"""Deterministic integer arithmetic coding primitives.

The implementation uses the classic finite-precision range update with E1,
E2, and E3 renormalization.  Probability tables and all coder state are
integers, so a fixed symbol/CDF sequence produces the same bytes on every
platform.
"""

from __future__ import annotations

from pointconstellation.bitstream_bits import BitReader, BitWriter

STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
MASK = FULL_RANGE - 1
HALF = FULL_RANGE >> 1
QUARTER = HALF >> 1
THREE_QUARTERS = QUARTER * 3
MAX_TOTAL = QUARTER


def _validate_interval(low_count: int, high_count: int, total: int) -> None:
    if not 0 <= low_count < high_count <= total:
        raise ValueError("invalid arithmetic-coder frequency interval")
    if total > MAX_TOTAL:
        raise ValueError("arithmetic-coder frequency total is too large")


class RangeEncoder:
    """Encode integer cumulative-frequency intervals into a byte string."""

    def __init__(self) -> None:
        self._low = 0
        self._high = MASK
        self._pending = 0
        self._writer = BitWriter()
        self._finished = False

    def _emit(self, bit: int) -> None:
        self._writer.write(bit, 1)
        opposite = bit ^ 1
        for _ in range(self._pending):
            self._writer.write(opposite, 1)
        self._pending = 0

    def encode(self, low_count: int, high_count: int, total: int) -> None:
        """Encode one symbol interval ``[low_count, high_count)``."""

        if self._finished:
            raise RuntimeError("arithmetic encoder has already been finished")
        _validate_interval(low_count, high_count, total)
        width = self._high - self._low + 1
        self._high = self._low + width * high_count // total - 1
        self._low = self._low + width * low_count // total
        while True:
            if self._high < HALF:
                self._emit(0)
            elif self._low >= HALF:
                self._emit(1)
                self._low -= HALF
                self._high -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTERS:
                self._pending += 1
                self._low -= QUARTER
                self._high -= QUARTER
            else:
                break
            self._low = (self._low << 1) & MASK
            self._high = ((self._high << 1) & MASK) | 1

    def finish(self) -> bytes:
        """Terminate the stream and return its canonical byte representation."""

        if self._finished:
            raise RuntimeError("arithmetic encoder has already been finished")
        self._finished = True
        self._pending += 1
        self._emit(0 if self._low < QUARTER else 1)
        return self._writer.finish()


class RangeDecoder:
    """Decode integer cumulative-frequency intervals from a byte string."""

    def __init__(self, data: bytes) -> None:
        if not data:
            raise ValueError("truncated arithmetic-coded payload")
        self._reader = BitReader(data, pad_with_zeros=True)
        self._low = 0
        self._high = MASK
        self._code = 0
        for _ in range(STATE_BITS):
            self._code = (self._code << 1) | self._reader.read(1)

    def cumulative(self, total: int) -> int:
        """Return the cumulative-frequency value for the next symbol."""

        if not 1 <= total <= MAX_TOTAL:
            raise ValueError("invalid arithmetic-coder frequency total")
        width = self._high - self._low + 1
        value = ((self._code - self._low + 1) * total - 1) // width
        if not 0 <= value < total:
            raise ValueError("corrupt arithmetic-coded payload")
        return value

    def decode(self, low_count: int, high_count: int, total: int) -> None:
        """Consume one decoded symbol interval."""

        _validate_interval(low_count, high_count, total)
        width = self._high - self._low + 1
        self._high = self._low + width * high_count // total - 1
        self._low = self._low + width * low_count // total
        while True:
            if self._high < HALF:
                pass
            elif self._low >= HALF:
                self._low -= HALF
                self._high -= HALF
                self._code -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTERS:
                self._low -= QUARTER
                self._high -= QUARTER
                self._code -= QUARTER
            else:
                break
            self._low = (self._low << 1) & MASK
            self._high = ((self._high << 1) & MASK) | 1
            self._code = ((self._code << 1) & MASK) | self._reader.read(1)
