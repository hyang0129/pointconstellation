"""Experiment 040 codec providers for the Experiment 041 anomaly benchmark.

Every encoder in this module accepts source coordinates only.  Learned arms
share a sealed Experiment 019 decoder and Experiment 040's FPS-start Adam-STE
search.  G-PCC freshly encodes the same source and selects its nearest measured
geometry-brick payload rather than reusing an older reconstruction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pointconstellation.bitstream import (
    HEADER,
    MODE_FIXED,
    SELECTIVE_HEADER,
    decode_constellation,
    encode_constellation,
    expected_payload_bytes,
)
from pointconstellation.codecs import GpccResult, Tmc3RatePoint, run_tmc3
from pointconstellation.defect_anomaly_benchmark import (
    CodecArm,
    DefectAnomalyBenchmarkConfig,
)
from pointconstellation.irregularity import (
    decoder_residual_score,
    deterministic_random_scores,
    local_geometry_scores,
    select_spaced_indices,
)
from pointconstellation.selective_codec import (
    decode_selective_message,
    encode_selective_message,
)
from pointconstellation.selective_experiment import (
    SCORE_METHODS,
    FrozenExperiment040CodecContext,
    SelectiveExperimentConfig,
    _gpcc_rate_points,
    _gpcc_reference_rows,
    _split_counts,
)

FloatArray = NDArray[np.float32]
DoubleArray = NDArray[np.float64]
_UNIFORM_PADDING_BYTES = SELECTIVE_HEADER.size - HEADER.size


def _source_points(
    source: FloatArray, *, expected_point_count: int | None = None
) -> FloatArray:
    values = np.asarray(source, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("codec source must have shape (N, 3) with N > 0")
    if not np.isfinite(values).all():
        raise ValueError("codec source must contain finite coordinates")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("codec source must lie in [-1, 1]")
    if expected_point_count is not None and len(values) != expected_point_count:
        raise ValueError(
            "codec source cardinality differs from the declared Experiment 041 "
            f"regime: expected N={expected_point_count}, observed N={len(values)}"
        )
    return values


def _canonical_source(
    source: FloatArray, *, expected_point_count: int | None = None
) -> FloatArray:
    values = _source_points(source, expected_point_count=expected_point_count)
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    return np.ascontiguousarray(values[order])


def _source_digest(source: FloatArray) -> str:
    return hashlib.sha256(_canonical_source(source).astype(">f4").tobytes()).hexdigest()


def _maximum_points(payload_budget: int, bits: int) -> int:
    count = (8 * payload_budget) // (3 * bits)
    while count and expected_payload_bytes(count, bits) > payload_budget:
        count -= 1
    if count < 1:
        raise ValueError("payload budget cannot hold one coordinate point")
    return count


def _rate_metadata(
    *,
    stream_bytes: int,
    header_bytes: int,
    payload_bytes: int,
    payload_budget_bytes: int,
    target_bytes: int,
    rate_point: str | None = None,
) -> dict[str, Any]:
    if header_bytes + payload_bytes != stream_bytes:
        raise RuntimeError("codec header and payload bytes do not sum to the stream")
    return {
        "codec_header_bytes": header_bytes,
        "codec_payload_bytes": payload_bytes,
        "payload_byte_delta": payload_bytes - payload_budget_bytes,
        "complete_stream_byte_delta": stream_bytes - target_bytes,
        "codec_rate_point": rate_point,
    }


class _ProviderState:
    """Shared frozen-decoder searches and source-local geometry caches."""

    def __init__(
        self,
        config: DefectAnomalyBenchmarkConfig,
        experiment_040: SelectiveExperimentConfig,
        *,
        device_name: str,
    ) -> None:
        self.config = config
        self.experiment_040 = experiment_040
        self.context = FrozenExperiment040CodecContext(
            experiment_040,
            decoder_seed=config.experiment_040_decoder_seed,
            device_name=device_name,
        )
        self.output_points = config.num_points
        self.decoder_maximum_output_points = self.context.stability.num_points
        if self.output_points > self.decoder_maximum_output_points:
            raise ValueError(
                "Experiment 041 regime exceeds the sealed decoder maximum: "
                f"N={self.output_points}, maximum={self.decoder_maximum_output_points}"
            )
        self._searches: dict[tuple[str, int, int], DoubleArray] = {}
        self._base_decodes: dict[tuple[str, int, int], DoubleArray] = {}
        self._geometry: dict[str, Any] = {}

    def search(self, source: FloatArray, *, size: int, bits: int) -> DoubleArray:
        values = _canonical_source(
            source, expected_point_count=self.output_points
        )
        digest = _source_digest(values)
        key = (digest, size, bits)
        if key not in self._searches:
            self._searches[key] = self.context.search(
                values,
                constellation_size=size,
                bits=bits,
                output_points=self.output_points,
            )
        return self._searches[key].copy()

    def decode(self, coordinates: DoubleArray, output_points: int) -> DoubleArray:
        if output_points != self.output_points:
            raise ValueError(
                "decoder request differs from the declared Experiment 041 regime: "
                f"expected N={self.output_points}, observed N={output_points}"
            )
        return self.context.decode(coordinates, output_points)

    def base_decode(
        self, source: FloatArray, *, size: int, bits: int
    ) -> DoubleArray:
        digest = _source_digest(source)
        key = (digest, size, bits)
        if key not in self._base_decodes:
            self._base_decodes[key] = self.decode(
                self.search(source, size=size, bits=bits), self.output_points
            )
        return self._base_decodes[key].copy()

    def score(
        self,
        source: FloatArray,
        *,
        base_reconstruction: DoubleArray,
        method: str,
        payload_budget: int,
        preserved_fraction: float,
    ) -> DoubleArray:
        values = _source_points(
            source, expected_point_count=self.output_points
        ).astype(np.float64)
        if method == "decoder_residual":
            return decoder_residual_score(
                values,
                base_reconstruction,
                chunk_size=self.experiment_040.irregularity_chunk_size,
            )
        if method == "random":
            identity = json.dumps(
                [
                    self.experiment_040.selection_seed,
                    _source_digest(source),
                    payload_budget,
                    preserved_fraction,
                ],
                separators=(",", ":"),
            )
            seed = int.from_bytes(
                hashlib.sha256(identity.encode()).digest()[:8], "big"
            ) % (2**63)
            return deterministic_random_scores(values, seed=seed)
        digest = _source_digest(source)
        if digest not in self._geometry:
            neighbors = min(
                self.experiment_040.irregularity_neighbors, len(values) - 1
            )
            self._geometry[digest] = local_geometry_scores(
                values,
                neighbors=neighbors,
                chunk_size=self.experiment_040.irregularity_chunk_size,
            )
        geometry = self._geometry[digest]
        if method == "curvature":
            return geometry.curvature
        if method == "density":
            return geometry.density_deviation
        if method == "boundary":
            return geometry.boundary
        raise ValueError(f"unknown Experiment 040 selection score: {method}")


class _ConstellationCodec:
    def __init__(
        self,
        state: _ProviderState,
        *,
        size: int,
        bits: int,
        payload_budget: int,
        target_bytes: int,
    ) -> None:
        self.state = state
        self.size = size
        self.bits = bits
        self.payload_budget = payload_budget
        self.target_bytes = target_bytes
        self.payload_bytes = expected_payload_bytes(size, bits)

    def encode(self, source: FloatArray) -> bytes:
        values = _canonical_source(
            source, expected_point_count=self.state.output_points
        )
        coordinates = self.state.search(values, size=self.size, bits=self.bits)
        canonical = encode_constellation(
            coordinates,
            bits=self.bits,
            mode=MODE_FIXED,
            output_points=self.state.output_points,
        )
        stream = canonical + bytes(_UNIFORM_PADDING_BYTES)
        if len(stream) != self.target_bytes:
            raise RuntimeError("padded constellation stream differs from target bytes")
        return stream

    def decode(self, stream: bytes) -> FloatArray:
        if len(stream) != self.target_bytes:
            raise ValueError("constellation stream differs from target bytes")
        if any(stream[-_UNIFORM_PADDING_BYTES:]):
            raise ValueError("constellation rate-matching padding is nonzero")
        packet = decode_constellation(stream[:-_UNIFORM_PADDING_BYTES])
        if packet.output_points != self.state.output_points:
            raise ValueError("constellation stream output count differs from regime")
        return self.state.decode(packet.coordinates, packet.output_points).astype(
            np.float32
        )

    def rate_metadata(self, stream: bytes) -> dict[str, Any]:
        return _rate_metadata(
            stream_bytes=len(stream),
            header_bytes=SELECTIVE_HEADER.size,
            payload_bytes=self.payload_bytes,
            payload_budget_bytes=self.payload_budget,
            target_bytes=self.target_bytes,
        )


class _SelectiveCodec:
    def __init__(
        self,
        state: _ProviderState,
        *,
        k1: int,
        k2: int,
        bits: int,
        payload_budget: int,
        target_bytes: int,
        score_method: str,
        preserved_fraction: float,
    ) -> None:
        self.state = state
        self.k1 = k1
        self.k2 = k2
        self.bits = bits
        self.payload_budget = payload_budget
        self.target_bytes = target_bytes
        self.score_method = score_method
        self.preserved_fraction = preserved_fraction
        self.payload_bytes = expected_payload_bytes(k1 + k2, bits)

    def encode(self, source: FloatArray) -> bytes:
        values = _canonical_source(
            source, expected_point_count=self.state.output_points
        )
        coordinates = self.state.search(values, size=self.k1, bits=self.bits)
        base = self.state.base_decode(values, size=self.k1, bits=self.bits)
        scores = self.state.score(
            values,
            base_reconstruction=base,
            method=self.score_method,
            payload_budget=self.payload_budget,
            preserved_fraction=self.preserved_fraction,
        )
        indices = select_spaced_indices(
            values,
            scores,
            self.k2,
            minimum_spacing=self.state.experiment_040.minimum_spacing,
        )
        stream = encode_selective_message(
            coordinates,
            values[indices],
            bits=self.bits,
            output_points=self.state.output_points,
        )
        if len(stream) != self.target_bytes:
            raise RuntimeError("selective stream differs from target bytes")
        return stream

    def decode(self, stream: bytes) -> FloatArray:
        if len(stream) != self.target_bytes:
            raise ValueError("selective stream differs from target bytes")
        decoded = decode_selective_message(stream, decoder=self.state.decode)
        if decoded.packet.output_points != self.state.output_points:
            raise ValueError("selective stream output count differs from regime")
        return decoded.reconstruction.astype(np.float32)

    def rate_metadata(self, stream: bytes) -> dict[str, Any]:
        return _rate_metadata(
            stream_bytes=len(stream),
            header_bytes=SELECTIVE_HEADER.size,
            payload_bytes=self.payload_bytes,
            payload_budget_bytes=self.payload_budget,
            target_bytes=self.target_bytes,
        )


@dataclass(frozen=True)
class _GpccCell:
    rate_point: Tmc3RatePoint
    result: GpccResult
    stream: bytes


class _GpccFrontier:
    """One freshly encoded TMC13 frontier per source coordinate set."""

    def __init__(
        self,
        config: DefectAnomalyBenchmarkConfig,
        experiment_040: SelectiveExperimentConfig,
    ) -> None:
        if experiment_040.gpcc_reference_path is None:
            raise ValueError("Experiment 041's G-PCC arm requires a reference grid")
        self.config = config
        self.experiment_040 = experiment_040
        rows = _gpcc_reference_rows(Path(experiment_040.gpcc_reference_path))
        self.rate_points = _gpcc_rate_points(rows)
        self.executable = Path(experiment_040.tmc3_executable)
        self.work_root = Path(config.output_dir) / "codec_scratch" / "gpcc"
        self._frontiers: dict[str, tuple[_GpccCell, ...]] = {}
        self._decoded: dict[str, DoubleArray] = {}
        self._metadata: dict[tuple[str, int], dict[str, Any]] = {}

    def _frontier(self, source: FloatArray) -> tuple[_GpccCell, ...]:
        values = _canonical_source(
            source, expected_point_count=self.config.num_points
        )
        digest = _source_digest(values)
        if digest not in self._frontiers:
            cells = []
            for rate_point in self.rate_points:
                work_dir = self.work_root / digest / rate_point.name
                result = run_tmc3(
                    self.executable,
                    values,
                    rate_point=rate_point,
                    work_dir=work_dir,
                    position_bits=self.experiment_040.position_bits,
                    timeout_seconds=self.experiment_040.timeout_seconds,
                )
                stream = (work_dir / "stream.bin").read_bytes()
                if result.stream_breakdown is None:
                    raise RuntimeError("TMC13 result lacks byte-exact accounting")
                if len(stream) != result.stream_breakdown.total_bytes:
                    raise RuntimeError("TMC13 stream differs from parsed byte total")
                stream_digest = hashlib.sha256(stream).hexdigest()
                reconstruction = result.reconstruction.astype(np.float64)
                previous = self._decoded.get(stream_digest)
                if previous is not None and not np.array_equal(
                    previous, reconstruction
                ):
                    raise RuntimeError("one TMC13 stream mapped to two reconstructions")
                self._decoded[stream_digest] = reconstruction
                cells.append(_GpccCell(rate_point, result, stream))
            self._frontiers[digest] = tuple(cells)
        return self._frontiers[digest]

    def encode(
        self, source: FloatArray, *, payload_budget: int, target_bytes: int
    ) -> bytes:
        cell = min(
            self._frontier(source),
            key=lambda item: (
                abs(item.result.stream_breakdown.payload_bytes - payload_budget),
                item.result.stream_bytes,
                item.rate_point.name,
            ),
        )
        breakdown = cell.result.stream_breakdown
        metadata = _rate_metadata(
            stream_bytes=len(cell.stream),
            header_bytes=breakdown.header_bytes,
            payload_bytes=breakdown.payload_bytes,
            payload_budget_bytes=payload_budget,
            target_bytes=target_bytes,
            rate_point=cell.rate_point.name,
        )
        metadata["fresh_source_encode"] = True
        metadata["matched_by"] = "nearest_actual_geometry_brick_payload"
        self._metadata[(hashlib.sha256(cell.stream).hexdigest(), payload_budget)] = (
            metadata
        )
        return cell.stream

    def decode(self, stream: bytes) -> FloatArray:
        digest = hashlib.sha256(stream).hexdigest()
        if digest not in self._decoded:
            raise ValueError("G-PCC decode requires a stream returned by encode")
        return self._decoded[digest].copy().astype(np.float32)

    def metadata(self, stream: bytes, *, payload_budget: int) -> dict[str, Any]:
        key = (hashlib.sha256(stream).hexdigest(), payload_budget)
        if key not in self._metadata:
            raise ValueError("G-PCC accounting requires a stream returned by encode")
        return dict(self._metadata[key])


class _GpccCodec:
    def __init__(
        self,
        frontier: _GpccFrontier,
        *,
        payload_budget: int,
        target_bytes: int,
    ) -> None:
        self.frontier = frontier
        self.payload_budget = payload_budget
        self.target_bytes = target_bytes

    def encode(self, source: FloatArray) -> bytes:
        return self.frontier.encode(
            source,
            payload_budget=self.payload_budget,
            target_bytes=self.target_bytes,
        )

    def decode(self, stream: bytes) -> FloatArray:
        return self.frontier.decode(stream)

    def rate_metadata(self, stream: bytes) -> dict[str, Any]:
        return self.frontier.metadata(stream, payload_budget=self.payload_budget)


def build_codec_arms(
    *, config: DefectAnomalyBenchmarkConfig, device_name: str
) -> tuple[CodecArm, ...]:
    """Build the real Experiment 040/041 arms at the shared payload ladder."""

    experiment_040 = SelectiveExperimentConfig.from_json(
        Path(config.experiment_040_config)
    )
    if config.experiment_040_decoder_seed not in experiment_040.decoder_seeds:
        raise ValueError("Experiment 041 decoder seed is absent from its 040 config")
    if config.selective_score_method not in SCORE_METHODS:
        raise ValueError("Experiment 041 score is not an Experiment 040 score")
    if config.selective_preserved_fraction is None:
        raise ValueError("real selective provider requires a preserved fraction")
    state = _ProviderState(config, experiment_040, device_name=device_name)
    gpcc = _GpccFrontier(config, experiment_040)
    arms = []
    bits = experiment_040.coordinate_bits
    for payload_budget in config.payload_budgets:
        total = _maximum_points(payload_budget, bits)
        k1, k2 = _split_counts(total, config.selective_preserved_fraction)
        payload_bytes = expected_payload_bytes(total, bits)
        target_bytes = SELECTIVE_HEADER.size + payload_bytes
        arms.extend(
            (
                CodecArm(
                    name=config.constellation_arm,
                    payload_budget_bytes=payload_budget,
                    target_bytes=target_bytes,
                    codec=_ConstellationCodec(
                        state,
                        size=total,
                        bits=bits,
                        payload_budget=payload_budget,
                        target_bytes=target_bytes,
                    ),
                    role="strict_constellation_fps_adam_ste",
                ),
                CodecArm(
                    name=config.selective_arm,
                    payload_budget_bytes=payload_budget,
                    target_bytes=target_bytes,
                    codec=_SelectiveCodec(
                        state,
                        k1=k1,
                        k2=k2,
                        bits=bits,
                        payload_budget=payload_budget,
                        target_bytes=target_bytes,
                        score_method=config.selective_score_method,
                        preserved_fraction=config.selective_preserved_fraction,
                    ),
                    role="experiment_040_selective_passthrough",
                ),
                CodecArm(
                    name=config.random_control_arm,
                    payload_budget_bytes=payload_budget,
                    target_bytes=target_bytes,
                    codec=_SelectiveCodec(
                        state,
                        k1=k1,
                        k2=k2,
                        bits=bits,
                        payload_budget=payload_budget,
                        target_bytes=target_bytes,
                        score_method="random",
                        preserved_fraction=config.selective_preserved_fraction,
                    ),
                    role="coordinate_keyed_random_k2_control",
                ),
                CodecArm(
                    name=config.gpcc_arm,
                    payload_budget_bytes=payload_budget,
                    target_bytes=target_bytes,
                    codec=_GpccCodec(
                        gpcc,
                        payload_budget=payload_budget,
                        target_bytes=target_bytes,
                    ),
                    exact_bytes=False,
                    maximum_rate_error_bytes=config.gpcc_maximum_rate_error_bytes,
                    role="fresh_tmc13_nearest_payload",
                ),
            )
        )
    return tuple(arms)


__all__ = ["build_codec_arms"]
