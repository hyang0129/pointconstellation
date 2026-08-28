"""Experiment 027 Draco arms with actual-rate and official-metric rows."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.codecs import DracoCodecSpec, run_draco, run_pc_error

SUBSET_METHODS = ("fps", "adam")
RECONSTRUCTION_ROUTES = ("direct", "frozen_decoder")


@dataclass(frozen=True)
class DracoArm:
    """One predeclared Draco geometry and reconstruction route."""

    name: str
    input_kind: str
    quantization_bits: int
    reconstruction_route: str = "direct"
    constellation_size: int | None = None

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("Draco arm name must be nonempty without whitespace")
        if self.input_kind not in {*SUBSET_METHODS, "full_cloud"}:
            raise ValueError("Draco input_kind must be fps, adam, or full_cloud")
        if self.reconstruction_route not in RECONSTRUCTION_ROUTES:
            raise ValueError("unknown Draco reconstruction route")
        if self.input_kind == "full_cloud":
            if self.constellation_size is not None:
                raise ValueError("full-cloud Draco cannot declare a constellation size")
            if self.reconstruction_route != "direct":
                raise ValueError("full-cloud Draco uses direct reconstruction")
            if self.quantization_bits not in range(3, 7):
                raise ValueError("full-cloud Draco qp must be in {3, 4, 5, 6}")
        else:
            if self.constellation_size is None or self.constellation_size < 1:
                raise ValueError("subset Draco requires a positive constellation size")
            if self.quantization_bits not in {8, 10, 12}:
                raise ValueError("subset Draco qp must be in {8, 10, 12}")


@dataclass(frozen=True)
class DracoBenchmarkConfig:
    """Checked protocol for the complete set of Experiment 027 Draco arms."""

    codec: DracoCodecSpec
    arms: tuple[DracoArm, ...]
    pc_error_executable: str
    metric_position_bits: int = 12
    metric_timeout_seconds: float = 120.0
    output_dir: str = "artifacts/local/experiment_027_draco"

    def __post_init__(self) -> None:
        if not self.arms or len({arm.name for arm in self.arms}) != len(self.arms):
            raise ValueError("Draco arms must be nonempty and uniquely named")
        if not 2 <= self.metric_position_bits <= 24:
            raise ValueError("metric_position_bits must be between 2 and 24")
        if self.metric_timeout_seconds <= 0:
            raise ValueError("metric_timeout_seconds must be positive")

    @classmethod
    def from_json(cls, path: Path) -> DracoBenchmarkConfig:
        values = json.loads(path.read_text())
        values["codec"] = DracoCodecSpec(**values["codec"])
        values["arms"] = tuple(DracoArm(**arm) for arm in values["arms"])
        return cls(**values)


def default_draco_arms(constellation_size: int) -> tuple[DracoArm, ...]:
    """Return the fixed subset, frozen-decoder, and high-rate arm list."""

    arms = []
    for method in SUBSET_METHODS:
        for bits in (8, 10, 12):
            for route in RECONSTRUCTION_ROUTES:
                arms.append(
                    DracoArm(
                        name=f"draco_{method}_k{constellation_size}_q{bits}_{route}",
                        input_kind=method,
                        quantization_bits=bits,
                        reconstruction_route=route,
                        constellation_size=constellation_size,
                    )
                )
    arms.extend(
        DracoArm(
            name=f"draco_full_qp{bits}",
            input_kind="full_cloud",
            quantization_bits=bits,
        )
        for bits in range(3, 7)
    )
    return tuple(arms)


def fps_subset(points: ArrayLike, constellation_size: int) -> NDArray[np.float32]:
    """Match the deterministic centroid-start FPS policy used by the refiner."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("FPS input must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all():
        raise ValueError("FPS input must be finite")
    if not 1 <= constellation_size <= len(array):
        raise ValueError("constellation_size must be between one and N")
    farthest = int(np.argmax(np.sum((array - array.mean(axis=0)) ** 2, axis=1)))
    minimum = np.full(len(array), np.inf)
    selected = np.empty(constellation_size, dtype=np.int64)
    for index in range(constellation_size):
        selected[index] = farthest
        distances = np.sum((array - array[farthest]) ** 2, axis=1)
        minimum = np.minimum(minimum, distances)
        farthest = int(np.argmax(minimum))
    return array[selected].astype(np.float32)


def evaluate_draco_arm(
    config: DracoBenchmarkConfig,
    arm: DracoArm,
    source: ArrayLike,
    normals: ArrayLike,
    *,
    work_dir: Path,
    split: str,
    family: str,
    model_id: str,
    sample_id: int,
    adam_constellation: ArrayLike | None = None,
    frozen_decoder: Callable[[NDArray[np.float32]], ArrayLike] | None = None,
    frozen_decoder_model_bytes: int | None = None,
) -> dict[str, Any]:
    """Evaluate one arm without importing the Torch-owned model implementation."""

    source_array = np.asarray(source, dtype=np.float32)
    normal_array = np.asarray(normals, dtype=np.float32)
    if source_array.shape != normal_array.shape:
        raise ValueError("source points and normals must have matching shapes")
    if arm.input_kind == "full_cloud":
        codec_input = source_array
    elif arm.input_kind == "fps":
        assert arm.constellation_size is not None
        codec_input = fps_subset(source_array, arm.constellation_size)
    else:
        if adam_constellation is None:
            raise ValueError("Adam Draco arm requires a selected constellation")
        codec_input = np.asarray(adam_constellation, dtype=np.float32)
        if codec_input.shape != (arm.constellation_size, 3):
            raise ValueError("Adam constellation shape differs from the declared K")

    codec = run_draco(
        config.codec,
        codec_input,
        quantization_bits=arm.quantization_bits,
        work_dir=work_dir / "codec",
    )
    model_bytes = 0
    if arm.reconstruction_route == "frozen_decoder":
        if frozen_decoder is None:
            raise ValueError("frozen-decoder Draco arm requires a decoder callback")
        if frozen_decoder_model_bytes is None or frozen_decoder_model_bytes < 1:
            raise ValueError("frozen-decoder model size must count deployment bytes")
        reconstruction = np.asarray(
            frozen_decoder(codec.reconstruction), dtype=np.float32
        )
        model_bytes = frozen_decoder_model_bytes
    else:
        reconstruction = codec.reconstruction
    with tempfile.TemporaryDirectory(prefix="metric-", dir=work_dir) as temporary:
        metric = run_pc_error(
            Path(config.pc_error_executable),
            source_array,
            reconstruction,
            normal_array,
            work_dir=Path(temporary),
            position_bits=config.metric_position_bits,
            timeout_seconds=config.metric_timeout_seconds,
        )
    return {
        "experiment": "027_draco",
        "dataset_split": split,
        "split": split,
        "family": family,
        "model_id": model_id,
        "sample_id": sample_id,
        "codec_family": "draco",
        "method": arm.name,
        "selection_method": arm.input_kind,
        "reconstruction_route": arm.reconstruction_route,
        "constellation_size": arm.constellation_size,
        "source_points": len(source_array),
        "codec_input_points": len(codec_input),
        "decoded_points": len(codec.reconstruction),
        "reconstruction_points": len(reconstruction),
        "draco_quantization_bits": arm.quantization_bits,
        "metric_position_bits": config.metric_position_bits,
        "payload_bytes": codec.payload_bytes,
        "stream_bytes": codec.payload_bytes,
        "actual_stream_bpp": 8.0 * codec.payload_bytes / len(source_array),
        "model_bytes": model_bytes,
        "stream_sha256": codec.stream_sha256,
        "reconstruction_sha256": codec.reconstruction_sha256,
        "encoder_sha256": codec.encoder_sha256,
        "decoder_sha256": codec.decoder_sha256,
        "encode_seconds": codec.encode_seconds,
        "decode_seconds": codec.decode_seconds,
        "official_metric_seconds": metric.elapsed_seconds,
        **metric.metrics,
    }
