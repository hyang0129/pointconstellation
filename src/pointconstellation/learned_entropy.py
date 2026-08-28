"""Shared probability models for learned constellation stream mode 2.

Two candidates are implemented.  ``octree`` codes child occupancy with a
training-seeded, stream-adaptive binary context model.  ``autoregressive`` is
a small fixed-point MLP that predicts each next lexicographically sorted
coordinate and codes its residual with an integer discretized-logistic table.
Only integer weights and probabilities are used by the encoder and decoder.
The fitted arrays are shared model state, never part of a per-cloud stream.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.range_coder import RangeDecoder, RangeEncoder

OCTREE = "octree"
AUTOREGRESSIVE = "autoregressive"
CANDIDATES = (OCTREE, AUTOREGRESSIVE)
MODEL_VERSION = 1
WEIGHT_SCALE = 1 << 16


@dataclass(frozen=True)
class LearnedEntropyConfig:
    """Deterministic fitting and integer-probability configuration."""

    candidate: str = OCTREE
    context_buckets: int = 8
    context_weight: int = 4
    smoothing: int = 1
    adaptive_increment: int = 1
    hidden_width: int = 24
    ridge: float = 1e-3
    logistic_frequency_total: int = 1 << 16
    seed: int = 20_260_826

    def __post_init__(self) -> None:
        if self.candidate not in CANDIDATES:
            raise ValueError(f"unknown learned entropy candidate: {self.candidate}")
        integer_values = (
            self.context_buckets,
            self.context_weight,
            self.smoothing,
            self.adaptive_increment,
            self.hidden_width,
            self.logistic_frequency_total,
        )
        if min(integer_values) < 1:
            raise ValueError("learned entropy integer settings must be positive")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if self.logistic_frequency_total > 1 << 20:
            raise ValueError("logistic_frequency_total cannot exceed 2^20")

    @classmethod
    def from_json(cls, path: Path) -> LearnedEntropyConfig:
        return cls(**json.loads(path.read_text()))


def _empty(dtype: np.dtype[Any] | None = None) -> NDArray[Any]:
    if dtype is None:
        dtype = np.dtype(np.int32)
    return np.empty((0,), dtype=dtype)


@dataclass(frozen=True)
class LearnedEntropyModel:
    """Integer inference state shared by mode-2 encoders and decoders."""

    config: LearnedEntropyConfig
    bits: int
    constellation_size: int
    octree_counts: NDArray[np.int32]
    input_weights: NDArray[np.int32]
    output_bias: NDArray[np.int64]
    output_weights: NDArray[np.int64]
    logistic_weights: NDArray[np.int32]
    training_streams: int = 0
    training_split: str = "untrained"

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 24:
            raise ValueError("model bits must be between 2 and 24")
        if not 1 <= self.constellation_size <= 65535:
            raise ValueError("model constellation_size must fit the stream")
        if self.training_streams < 0:
            raise ValueError("training_streams cannot be negative")
        if self.config.candidate == OCTREE:
            expected = (
                self.bits,
                min(self.constellation_size, 8) + 1,
                self.config.context_buckets + 1,
                8,
                9,
                2,
            )
            if self.octree_counts.shape != expected:
                raise ValueError(
                    f"octree count shape {self.octree_counts.shape} != {expected}"
                )
        else:
            symbols = 3 * self.constellation_size
            template_width = 2 * ((1 << self.bits) - 1) + 1
            if self.input_weights.shape != (symbols + 1, self.config.hidden_width):
                raise ValueError("autoregressive input weight shape is invalid")
            if self.output_bias.shape != (symbols,):
                raise ValueError("autoregressive bias shape is invalid")
            if self.output_weights.shape != (symbols, self.config.hidden_width):
                raise ValueError("autoregressive output weight shape is invalid")
            if self.logistic_weights.shape != (symbols, template_width):
                raise ValueError("autoregressive logistic table shape is invalid")
        for array in self.arrays:
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError("learned entropy inference arrays must be integers")

    @property
    def arrays(self) -> tuple[NDArray[Any], ...]:
        return (
            self.octree_counts,
            self.input_weights,
            self.output_bias,
            self.output_weights,
            self.logistic_weights,
        )

    @property
    def parameter_bytes(self) -> int:
        """Return uncompressed shared integer-array storage, excluding metadata."""

        return sum(array.nbytes for array in self.arrays)

    @property
    def model_hash(self) -> str:
        """Return a canonical hash of metadata, shapes, dtypes, and weights."""

        digest = hashlib.sha256()
        digest.update(self._metadata(include_hash=False).encode())
        for array in self.arrays:
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(json.dumps(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def _metadata(self, *, include_hash: bool) -> str:
        values: dict[str, Any] = {
            "version": MODEL_VERSION,
            "config": asdict(self.config),
            "bits": self.bits,
            "constellation_size": self.constellation_size,
            "training_streams": self.training_streams,
            "training_split": self.training_split,
        }
        if include_hash:
            values["model_hash"] = self.model_hash
        return json.dumps(values, sort_keys=True, separators=(",", ":"))

    def to_bytes(self) -> bytes:
        """Serialize the shared model to a deterministic NPZ payload."""

        output = io.BytesIO()
        np.savez_compressed(
            output,
            metadata=np.asarray(self._metadata(include_hash=True)),
            octree_counts=self.octree_counts,
            input_weights=self.input_weights,
            output_bias=self.output_bias,
            output_weights=self.output_weights,
            logistic_weights=self.logistic_weights,
        )
        return output.getvalue()

    def save(self, path: Path) -> None:
        path.write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: Path) -> LearnedEntropyModel:
        with np.load(path, allow_pickle=False) as values:
            metadata = json.loads(str(values["metadata"].item()))
            if metadata.pop("version") != MODEL_VERSION:
                raise ValueError("unsupported learned entropy model version")
            expected_hash = metadata.pop("model_hash")
            model = cls(
                config=LearnedEntropyConfig(**metadata["config"]),
                bits=int(metadata["bits"]),
                constellation_size=int(metadata["constellation_size"]),
                training_streams=int(metadata["training_streams"]),
                training_split=str(metadata["training_split"]),
                octree_counts=values["octree_counts"].copy(),
                input_weights=values["input_weights"].copy(),
                output_bias=values["output_bias"].copy(),
                output_weights=values["output_weights"].copy(),
                logistic_weights=values["logistic_weights"].copy(),
            )
        if model.model_hash != expected_hash:
            raise ValueError("learned entropy model hash mismatch")
        return model

    def validate_stream(self, bits: int, constellation_size: int) -> None:
        if bits != self.bits or constellation_size != self.constellation_size:
            raise ValueError(
                "learned entropy model does not match stream precision and K"
            )


@lru_cache(maxsize=32)
def default_octree_model(bits: int, constellation_size: int) -> LearnedEntropyModel:
    """Construct the deterministic untrained adaptive-octree candidate."""

    config = LearnedEntropyConfig(candidate=OCTREE)
    shape = (
        bits,
        min(constellation_size, 8) + 1,
        config.context_buckets + 1,
        8,
        9,
        2,
    )
    return LearnedEntropyModel(
        config=config,
        bits=bits,
        constellation_size=constellation_size,
        octree_counts=np.zeros(shape, dtype=np.int32),
        input_weights=_empty(),
        output_bias=_empty(np.dtype(np.int64)),
        output_weights=_empty(np.dtype(np.int64)),
        logistic_weights=_empty(),
    )


def _lattices(values: ArrayLike, bits: int) -> NDArray[np.uint32]:
    lattice = np.asarray(values)
    if lattice.ndim != 3 or lattice.shape[2] != 3 or not len(lattice):
        raise ValueError("training lattices must have shape (C, K, 3)")
    levels = (1 << bits) - 1
    if not np.issubdtype(lattice.dtype, np.integer):
        raise ValueError("training lattices must contain integer coordinates")
    if np.any(lattice < 0) or np.any(lattice > levels):
        raise ValueError("training coordinate lies outside the model lattice")
    result = lattice.astype(np.uint32)
    return np.stack(
        [row[np.lexsort((row[:, 2], row[:, 1], row[:, 0]))] for row in result]
    )


def fit_learned_entropy_model(
    lattices: ArrayLike,
    *,
    bits: int,
    config: LearnedEntropyConfig,
    training_split: str = "train",
) -> LearnedEntropyModel:
    """Fit one candidate using only the caller-supplied training lattices."""

    values = _lattices(lattices, bits)
    if training_split != "train":
        raise ValueError("learned entropy models must be fitted on split=train")
    if config.candidate == OCTREE:
        return _fit_octree(values, bits=bits, config=config)
    return _fit_autoregressive(values, bits=bits, config=config)


def _context_bucket(prefix: int, depth: int, buckets: int) -> int:
    mixed = prefix * 0x9E3779B1 + (depth + 1) * 0x85EBCA77
    return 1 + mixed % buckets


def _octree_observations(
    lattice: NDArray[np.uint32], bits: int
) -> list[tuple[int, int, int, int, int, int]]:
    observations = []
    queue: list[tuple[NDArray[np.uint32], int]] = [(lattice, 0)]
    for depth in range(bits):
        shift = bits - depth - 1
        next_queue = []
        for points, prefix in queue:
            child_ids = (
                ((points[:, 0] >> shift) & 1) << 2
                | ((points[:, 1] >> shift) & 1) << 1
                | ((points[:, 2] >> shift) & 1)
            ).astype(np.int64)
            occupied = 0
            for child in range(8):
                selected = points[child_ids == child]
                bit = int(len(selected) > 0)
                observations.append((depth, len(points), prefix, child, occupied, bit))
                occupied += bit
                if bit:
                    next_queue.append((selected, (prefix << 3) | child))
        queue = next_queue
    return observations


def _fit_octree(
    lattices: NDArray[np.uint32],
    *,
    bits: int,
    config: LearnedEntropyConfig,
) -> LearnedEntropyModel:
    size = lattices.shape[1]
    counts = np.zeros(
        (
            bits,
            min(size, 8) + 1,
            config.context_buckets + 1,
            8,
            9,
            2,
        ),
        dtype=np.int32,
    )
    for lattice in lattices:
        for depth, points, prefix, child, occupied, bit in _octree_observations(
            lattice, bits
        ):
            point_bucket = min(points, 8)
            context = _context_bucket(prefix, depth, config.context_buckets)
            counts[depth, point_bucket, 0, child, occupied, bit] += 1
            counts[depth, point_bucket, context, child, occupied, bit] += 1
    return LearnedEntropyModel(
        config=config,
        bits=bits,
        constellation_size=size,
        octree_counts=counts,
        input_weights=_empty(),
        output_bias=_empty(np.dtype(np.int64)),
        output_weights=_empty(np.dtype(np.int64)),
        logistic_weights=_empty(),
        training_streams=len(lattices),
        training_split="train",
    )


def _hidden_features(
    values: NDArray[np.int64], position: int, model: LearnedEntropyModel
) -> NDArray[np.int64]:
    levels = (1 << model.bits) - 1
    features = np.zeros(3 * model.constellation_size + 1, dtype=np.int64)
    features[0] = levels
    features[1 : position + 1] = values[:position] - levels // 2
    projection = features @ model.input_weights.astype(np.int64)
    return np.maximum(projection, 0)


def _predicted_mean(
    values: NDArray[np.int64], position: int, model: LearnedEntropyModel
) -> int:
    levels = (1 << model.bits) - 1
    hidden = _hidden_features(values, position, model)
    hidden_scale = levels * 8
    adjustment = int(hidden @ model.output_weights[position]) // hidden_scale
    fixed_point = int(model.output_bias[position]) + adjustment
    prediction = (fixed_point + WEIGHT_SCALE // 2) // WEIGHT_SCALE
    return min(levels, max(0, prediction))


def _logistic_table(
    residuals: NDArray[np.int64], levels: int, total: int
) -> NDArray[np.int32]:
    centered = residuals.astype(np.float64) - np.median(residuals)
    initial = max(1.0, float(np.mean(np.abs(centered))))
    candidates = np.geomspace(max(0.5, initial / 3.0), initial * 3.0 + 1.0, 17)
    best_scale = float(candidates[0])
    best_nll = math.inf
    for scale in candidates:
        upper = np.clip((residuals + 0.5) / scale, -50.0, 50.0)
        lower = np.clip((residuals - 0.5) / scale, -50.0, 50.0)
        probability = 1.0 / (1.0 + np.exp(-upper)) - 1.0 / (1.0 + np.exp(-lower))
        nll = float(-np.log(np.clip(probability, 1e-15, None)).sum())
        if nll < best_nll:
            best_nll = nll
            best_scale = float(scale)
    offsets = np.arange(-levels, levels + 1, dtype=np.float64)
    upper = np.clip((offsets + 0.5) / best_scale, -50.0, 50.0)
    lower = np.clip((offsets - 0.5) / best_scale, -50.0, 50.0)
    probability = 1.0 / (1.0 + np.exp(-upper)) - 1.0 / (1.0 + np.exp(-lower))
    return np.maximum(1, np.rint(probability * total)).astype(np.int32)


def _fit_autoregressive(
    lattices: NDArray[np.uint32],
    *,
    bits: int,
    config: LearnedEntropyConfig,
) -> LearnedEntropyModel:
    streams = lattices.reshape(len(lattices), -1).astype(np.int64)
    symbols = streams.shape[1]
    levels = (1 << bits) - 1
    rng = np.random.default_rng(config.seed)
    input_weights = rng.integers(
        -4,
        5,
        size=(symbols + 1, config.hidden_width),
        dtype=np.int32,
    )
    output_bias = np.empty(symbols, dtype=np.int64)
    output_weights = np.empty((symbols, config.hidden_width), dtype=np.int64)
    logistic_weights = np.empty((symbols, 2 * levels + 1), dtype=np.int32)
    partial_model = LearnedEntropyModel(
        config=config,
        bits=bits,
        constellation_size=lattices.shape[1],
        octree_counts=_empty(),
        input_weights=input_weights,
        output_bias=np.zeros(symbols, dtype=np.int64),
        output_weights=np.zeros((symbols, config.hidden_width), dtype=np.int64),
        logistic_weights=np.ones((symbols, 2 * levels + 1), dtype=np.int32),
    )
    hidden_scale = levels * 8
    for position in range(symbols):
        hidden = np.stack(
            [_hidden_features(row, position, partial_model) for row in streams]
        ).astype(np.float64)
        design = np.column_stack((np.ones(len(hidden)), hidden / float(hidden_scale)))
        gram = design.T @ design
        gram.flat[:: len(gram) + 1] += config.ridge
        coefficients = np.linalg.solve(gram, design.T @ streams[:, position])
        output_bias[position] = int(np.rint(coefficients[0] * WEIGHT_SCALE))
        output_weights[position] = np.rint(coefficients[1:] * WEIGHT_SCALE).astype(
            np.int64
        )
    fitted = LearnedEntropyModel(
        config=config,
        bits=bits,
        constellation_size=lattices.shape[1],
        octree_counts=_empty(),
        input_weights=input_weights,
        output_bias=output_bias,
        output_weights=output_weights,
        logistic_weights=np.ones((symbols, 2 * levels + 1), dtype=np.int32),
    )
    for position in range(symbols):
        predictions = np.asarray(
            [_predicted_mean(row, position, fitted) for row in streams],
            dtype=np.int64,
        )
        residuals = streams[:, position] - predictions
        logistic_weights[position] = _logistic_table(
            residuals, levels, config.logistic_frequency_total
        )
    return LearnedEntropyModel(
        config=config,
        bits=bits,
        constellation_size=lattices.shape[1],
        octree_counts=_empty(),
        input_weights=input_weights,
        output_bias=output_bias,
        output_weights=output_weights,
        logistic_weights=logistic_weights,
        training_streams=len(lattices),
        training_split="train",
    )


def _binary_frequencies(
    model: LearnedEntropyModel,
    *,
    depth: int,
    points: int,
    prefix: int,
    child: int,
    occupied: int,
    local: dict[tuple[int, int, int, int, int], list[int]],
) -> tuple[int, int, tuple[int, int, int, int, int]]:
    point_bucket = min(points, 8)
    context = _context_bucket(prefix, depth, model.config.context_buckets)
    key = (depth, point_bucket, context, child, occupied)
    counts = model.octree_counts
    base = counts[depth, point_bucket, 0, child, occupied].astype(np.int64)
    contextual = counts[depth, point_bucket, context, child, occupied].astype(np.int64)
    adaptive = local.get(key, [0, 0])
    frequencies = (
        model.config.smoothing
        + base
        + model.config.context_weight * contextual
        + np.asarray(adaptive, dtype=np.int64)
    )
    return int(frequencies[0]), int(frequencies[1]), key


def _encode_octree(lattice: NDArray[np.uint32], model: LearnedEntropyModel) -> bytes:
    encoder = RangeEncoder()
    local: dict[tuple[int, int, int, int, int], list[int]] = {}
    queue: list[tuple[NDArray[np.uint32], int]] = [(lattice, 0)]
    for depth in range(model.bits):
        shift = model.bits - depth - 1
        next_queue = []
        for points, prefix in queue:
            child_ids = (
                ((points[:, 0] >> shift) & 1) << 2
                | ((points[:, 1] >> shift) & 1) << 1
                | ((points[:, 2] >> shift) & 1)
            ).astype(np.int64)
            children = [points[child_ids == child] for child in range(8)]
            occupied = 0
            for child, selected in enumerate(children):
                bit = int(len(selected) > 0)
                zero, one, key = _binary_frequencies(
                    model,
                    depth=depth,
                    points=len(points),
                    prefix=prefix,
                    child=child,
                    occupied=occupied,
                    local=local,
                )
                inferred = occupied == len(points) or (child == 7 and occupied == 0)
                if not inferred:
                    if bit:
                        encoder.encode(zero, zero + one, zero + one)
                    else:
                        encoder.encode(0, zero, zero + one)
                update = local.setdefault(key, [0, 0])
                update[bit] += model.config.adaptive_increment
                occupied += bit
            occupied_children = [child for child in children if len(child)]
            remaining = len(points)
            for index, selected in enumerate(occupied_children[:-1]):
                children_left = len(occupied_children) - index - 1
                maximum = remaining - children_left
                if maximum > 1:
                    encoder.encode(len(selected) - 1, len(selected), maximum)
                remaining -= len(selected)
            for child, selected in enumerate(children):
                if len(selected):
                    next_queue.append((selected, (prefix << 3) | child))
        queue = next_queue
    return encoder.finish()


def _decode_octree(data: bytes, model: LearnedEntropyModel) -> NDArray[np.uint32]:
    decoder = RangeDecoder(data)
    local: dict[tuple[int, int, int, int, int], list[int]] = {}
    queue: list[tuple[int, int]] = [(model.constellation_size, 0)]
    for depth in range(model.bits):
        next_queue = []
        for points, prefix in queue:
            mask = []
            occupied = 0
            for child in range(8):
                zero, one, key = _binary_frequencies(
                    model,
                    depth=depth,
                    points=points,
                    prefix=prefix,
                    child=child,
                    occupied=occupied,
                    local=local,
                )
                if occupied == points:
                    bit = 0
                elif child == 7 and occupied == 0:
                    bit = 1
                else:
                    cumulative = decoder.cumulative(zero + one)
                    bit = int(cumulative >= zero)
                    if bit:
                        decoder.decode(zero, zero + one, zero + one)
                    else:
                        decoder.decode(0, zero, zero + one)
                update = local.setdefault(key, [0, 0])
                update[bit] += model.config.adaptive_increment
                mask.append(bit)
                occupied += bit
            if not occupied or occupied > points:
                raise ValueError("corrupt learned octree occupancy")
            allocations = []
            remaining = points
            for index in range(occupied - 1):
                children_left = occupied - index - 1
                maximum = remaining - children_left
                if maximum > 1:
                    cumulative = decoder.cumulative(maximum)
                    count = cumulative + 1
                    decoder.decode(cumulative, cumulative + 1, maximum)
                else:
                    count = 1
                allocations.append(count)
                remaining -= count
            allocations.append(remaining)
            allocation_index = 0
            for child, bit in enumerate(mask):
                if bit:
                    next_queue.append(
                        (allocations[allocation_index], (prefix << 3) | child)
                    )
                    allocation_index += 1
        queue = next_queue
    lattice = []
    for count, prefix in queue:
        x = y = z = 0
        for depth in range(model.bits):
            child = (prefix >> (3 * (model.bits - depth - 1))) & 7
            x = (x << 1) | ((child >> 2) & 1)
            y = (y << 1) | ((child >> 1) & 1)
            z = (z << 1) | (child & 1)
        lattice.extend([(x, y, z)] * count)
    if len(lattice) != model.constellation_size:
        raise ValueError("corrupt learned octree point count")
    result = np.asarray(lattice, dtype=np.uint32)
    order = np.lexsort((result[:, 2], result[:, 1], result[:, 0]))
    return result[order]


def _canonical_bounds(
    values: NDArray[np.int64], position: int, levels: int
) -> tuple[int, int]:
    point, axis = divmod(position, 3)
    if point == 0:
        return 0, levels
    previous = values[(point - 1) * 3 : point * 3]
    current = values[point * 3 : position]
    if axis == 0:
        return int(previous[0]), levels
    if int(current[0]) != int(previous[0]):
        return 0, levels
    if axis == 1:
        return int(previous[1]), levels
    if int(current[1]) != int(previous[1]):
        return 0, levels
    return int(previous[2]), levels


def _autoregressive_interval(
    values: NDArray[np.int64],
    position: int,
    model: LearnedEntropyModel,
    symbol: int | None,
    cumulative: int | None,
) -> tuple[int, int, int, int]:
    levels = (1 << model.bits) - 1
    mean = _predicted_mean(values, position, model)
    lower, upper = _canonical_bounds(values, position, levels)
    template = model.logistic_weights[position]
    prefix = np.empty(len(template) + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(template, dtype=np.int64, out=prefix[1:])
    base_index = lower - mean + levels
    end_index = upper - mean + levels + 1
    base = int(prefix[base_index])
    total = int(prefix[end_index]) - base
    if symbol is None:
        assert cumulative is not None
        template_cumulative = base + cumulative
        index = int(np.searchsorted(prefix, template_cumulative, side="right") - 1)
        symbol = index + mean - levels
    if not lower <= symbol <= upper:
        raise ValueError("corrupt autoregressive canonical coordinate")
    index = symbol - mean + levels
    low_count = int(prefix[index]) - base
    high_count = int(prefix[index + 1]) - base
    return symbol, low_count, high_count, total


def _encode_autoregressive(
    lattice: NDArray[np.uint32], model: LearnedEntropyModel
) -> bytes:
    values = lattice.reshape(-1).astype(np.int64)
    encoder = RangeEncoder()
    for position, symbol_value in enumerate(values):
        _, low, high, total = _autoregressive_interval(
            values, position, model, int(symbol_value), None
        )
        encoder.encode(low, high, total)
    return encoder.finish()


def _decode_autoregressive(
    data: bytes, model: LearnedEntropyModel
) -> NDArray[np.uint32]:
    values = np.zeros(3 * model.constellation_size, dtype=np.int64)
    decoder = RangeDecoder(data)
    for position in range(len(values)):
        lower, _ = _canonical_bounds(values, position, (1 << model.bits) - 1)
        _, _, _, total = _autoregressive_interval(values, position, model, lower, None)
        cumulative = decoder.cumulative(total)
        symbol, low, high, _ = _autoregressive_interval(
            values, position, model, None, cumulative
        )
        decoder.decode(low, high, total)
        values[position] = symbol
    return values.reshape(-1, 3).astype(np.uint32)


def _checksum(header: bytes, lattice: NDArray[np.uint32]) -> int:
    canonical = np.ascontiguousarray(lattice.astype(">u4", copy=False)).tobytes()
    return zlib.crc32(header + canonical)


def encode_learned_lattice(
    lattice: NDArray[np.uint32],
    *,
    model: LearnedEntropyModel,
    header: bytes,
    minimum_payload_bytes: int = 0,
) -> bytes:
    """Encode one canonical lattice with shared model state and a CRC-32."""

    model.validate_stream(model.bits, len(lattice))
    if model.config.candidate == OCTREE:
        coded = _encode_octree(lattice, model)
    else:
        coded = _encode_autoregressive(lattice, model)
    checksum = _checksum(header, lattice).to_bytes(4, "big")
    padding = max(0, minimum_payload_bytes - len(coded) - len(checksum))
    return coded + bytes(padding) + checksum


def decode_learned_lattice(
    payload: bytes,
    *,
    model: LearnedEntropyModel,
    header: bytes,
) -> NDArray[np.uint32]:
    """Decode mode 2 and reject checksum, model, and arithmetic corruption."""

    if len(payload) < 5:
        raise ValueError("truncated learned constellation payload")
    coded = payload[:-4]
    expected_checksum = int.from_bytes(payload[-4:], "big")
    if model.config.candidate == OCTREE:
        lattice = _decode_octree(coded, model)
    else:
        lattice = _decode_autoregressive(coded, model)
    if _checksum(header, lattice) != expected_checksum:
        raise ValueError("learned constellation checksum mismatch")
    return lattice
