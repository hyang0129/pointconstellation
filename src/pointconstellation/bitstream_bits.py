"""Small MSB-first bit reader/writer shared by stream coders."""

from __future__ import annotations


class BitWriter:
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


class BitReader:
    def __init__(self, data: bytes, *, pad_with_zeros: bool = False) -> None:
        self._data = data
        self._pad_with_zeros = pad_with_zeros
        self.position = 0

    def read(self, width: int) -> int:
        if width < 0:
            raise ValueError("bit width cannot be negative")
        if not self._pad_with_zeros and self.position + width > 8 * len(self._data):
            raise ValueError("truncated constellation payload")
        value = 0
        for _ in range(width):
            if self.position >= 8 * len(self._data):
                bit = 0
            else:
                byte = self._data[self.position // 8]
                shift = 7 - self.position % 8
                bit = (byte >> shift) & 1
            value = (value << 1) | bit
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
        expected_bytes = (self.position + 7) // 8
        if len(self._data) != expected_bytes:
            raise ValueError("unexpected bytes after constellation payload")
        while self.position < 8 * len(self._data):
            if self.read(1):
                raise ValueError("non-zero constellation padding bits")
