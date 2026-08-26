"""Auditable subprocess adapter for Google Draco point-cloud streams."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.codecs.external import _read_ply_xyz
from pointconstellation.codecs.gpcc import write_ascii_ply

DRACO_RELEASE = "1.5.7"
DRACO_RELEASE_COMMIT = "8786740086a9f4d83f44aa83badfbea4dce7a1b5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executable(path: str, *, role: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"Draco {role} is missing or not executable: {path}")
    return resolved.resolve()


def _hash(value: str | None, *, field: str) -> None:
    if value is not None and (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True)
class DracoCodecSpec:
    """Pinned executables and managed Draco encoder options."""

    encoder_executable: str
    decoder_executable: str
    release: str = DRACO_RELEASE
    release_commit: str = DRACO_RELEASE_COMMIT
    encoder_sha256: str | None = None
    decoder_sha256: str | None = None
    compression_level: int = 10
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.release != DRACO_RELEASE:
            raise ValueError(f"Draco release must be pinned to {DRACO_RELEASE}")
        if self.release_commit != DRACO_RELEASE_COMMIT:
            raise ValueError("Draco release commit differs from the pinned release")
        _hash(self.encoder_sha256, field="encoder_sha256")
        _hash(self.decoder_sha256, field="decoder_sha256")
        if not 0 <= self.compression_level <= 10:
            raise ValueError("Draco compression_level must be between 0 and 10")
        if self.timeout_seconds <= 0:
            raise ValueError("Draco timeout_seconds must be positive")

    @classmethod
    def from_json(cls, path: Path) -> DracoCodecSpec:
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class DracoResult:
    """Measured output of an independently decoded Draco stream."""

    reconstruction: NDArray[np.float32]
    payload_bytes: int
    stream_sha256: str
    reconstruction_sha256: str
    encoder_sha256: str
    decoder_sha256: str
    encode_seconds: float
    decode_seconds: float
    encoder_command: tuple[str, ...]
    decoder_command: tuple[str, ...]
    encoder_output: str
    decoder_output: str

    @property
    def stream_bytes(self) -> int:
        """Compatibility alias used by existing actual-stream benchmarks."""

        return self.payload_bytes


def _points(points: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("Draco input must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all():
        raise ValueError("Draco input coordinates must be finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("Draco input coordinates must lie in [-1, 1]")
    return array


def _run(
    command: tuple[str, ...], *, timeout_seconds: float, stage: str
) -> tuple[str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(
            f"Draco {stage} failed with status {completed.returncode}: "
            f"{' '.join(command)}\n{output[-4000:]}"
        )
    return output, elapsed


def run_draco(
    spec: DracoCodecSpec,
    points: ArrayLike,
    *,
    quantization_bits: int,
    work_dir: Path,
) -> DracoResult:
    """Encode a point set, independently decode it, and count actual bytes."""

    if not 1 <= quantization_bits <= 30:
        raise ValueError("Draco quantization_bits must be between 1 and 30")
    array = _points(points)
    encoder = _executable(spec.encoder_executable, role="encoder")
    decoder = _executable(spec.decoder_executable, role="decoder")
    encoder_sha256 = _sha256(encoder)
    decoder_sha256 = _sha256(decoder)
    if spec.encoder_sha256 is not None and encoder_sha256 != spec.encoder_sha256:
        raise RuntimeError("Draco encoder SHA-256 differs from the declaration")
    if spec.decoder_sha256 is not None and decoder_sha256 != spec.decoder_sha256:
        raise RuntimeError("Draco decoder SHA-256 differs from the declaration")

    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.ply"
    stream_path = work_dir / "stream.drc"
    reconstruction_path = work_dir / "reconstruction.ply"
    if stream_path.exists() or reconstruction_path.exists():
        raise FileExistsError(
            "Draco work directory already contains output files; use a fresh "
            "per-run directory"
        )
    write_ascii_ply(input_path, array)
    encoder_command = (
        str(encoder),
        "-point_cloud",
        "-i",
        str(input_path.resolve()),
        "-o",
        str(stream_path.resolve()),
        "-qp",
        str(quantization_bits),
        "-cl",
        str(spec.compression_level),
    )
    encoder_output, encode_seconds = _run(
        encoder_command,
        timeout_seconds=spec.timeout_seconds,
        stage="encoder",
    )
    if not stream_path.is_file() or stream_path.stat().st_size < 1:
        raise RuntimeError("Draco encoder produced no nonempty stream")

    decoder_command = (
        str(decoder),
        "-i",
        str(stream_path.resolve()),
        "-o",
        str(reconstruction_path.resolve()),
    )
    decoder_output, decode_seconds = _run(
        decoder_command,
        timeout_seconds=spec.timeout_seconds,
        stage="decoder",
    )
    if not reconstruction_path.is_file():
        raise RuntimeError("Draco decoder produced no reconstruction")
    reconstruction = _read_ply_xyz(reconstruction_path)
    if not np.isfinite(reconstruction).all():
        raise RuntimeError("Draco decoded non-finite coordinates")
    return DracoResult(
        reconstruction=reconstruction,
        payload_bytes=stream_path.stat().st_size,
        stream_sha256=_sha256(stream_path),
        reconstruction_sha256=_sha256(reconstruction_path),
        encoder_sha256=encoder_sha256,
        decoder_sha256=decoder_sha256,
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        encoder_command=encoder_command,
        decoder_command=decoder_command,
        encoder_output=encoder_output,
        decoder_output=decoder_output,
    )
