"""Auditable subprocess adapter for MPEG G-PCC TMC13 geometry streams."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

RESERVED_OPTIONS = (
    "--mode",
    "--uncompressedDataPath",
    "--compressedStreamPath",
    "--reconstructedDataPath",
    "--frameCount",
    "--outputBinaryPly",
    "--disableAttributeCoding",
    "--mergeDuplicatedPoints",
)


@dataclass(frozen=True)
class Tmc3RatePoint:
    name: str
    encoder_args: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("rate-point name must be nonempty without whitespace")
        for argument in self.encoder_args:
            if not argument.startswith("--") or "=" not in argument:
                raise ValueError("TMC13 arguments must use --option=value syntax")
            if argument.startswith(RESERVED_OPTIONS):
                raise ValueError(
                    f"rate point cannot override managed option: {argument}"
                )


@dataclass(frozen=True)
class GpccResult:
    reconstruction: NDArray[np.float32]
    stream_bytes: int
    encode_seconds: float
    decode_seconds: float
    encoder_command: tuple[str, ...]
    decoder_command: tuple[str, ...]
    encoder_stdout: str
    decoder_stdout: str


@dataclass(frozen=True)
class OfficialMetricResult:
    metrics: dict[str, float]
    elapsed_seconds: float
    command: tuple[str, ...]
    stdout: str


def _points(points: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("point cloud must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all():
        raise ValueError("point cloud must contain finite coordinates")
    return array


def write_ascii_ply(path: Path, points: ArrayLike) -> None:
    array = _points(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(array)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("w") as handle:
        handle.write(header)
        np.savetxt(handle, array, fmt="%.9g %.9g %.9g")


def write_ascii_ply_with_normals(
    path: Path, points: ArrayLike, normals: ArrayLike
) -> None:
    point_array = _points(points)
    normal_array = _points(normals)
    if point_array.shape != normal_array.shape:
        raise ValueError("points and normals must have matching shapes")
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(point_array)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "end_header\n"
    )
    with path.open("w") as handle:
        handle.write(header)
        np.savetxt(
            handle,
            np.column_stack((point_array, normal_array)),
            fmt="%.9g %.9g %.9g %.9g %.9g %.9g",
        )


def read_ascii_ply(path: Path) -> NDArray[np.float32]:
    with path.open() as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"not a PLY file: {path}")
        if handle.readline().strip() != "format ascii 1.0":
            raise ValueError(f"only ASCII PLY 1.0 is supported: {path}")
        vertex_count: int | None = None
        properties: list[str] = []
        in_vertex = False
        for line in handle:
            fields = line.strip().split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields and fields[0] == "element":
                in_vertex = False
            elif fields[:1] == ["property"] and in_vertex:
                properties.append(fields[-1])
            elif fields[:1] == ["end_header"]:
                break
        else:
            raise ValueError(f"PLY header is unterminated: {path}")
        if vertex_count is None or vertex_count < 1:
            raise ValueError(f"PLY has no vertices: {path}")
        try:
            xyz = tuple(properties.index(axis) for axis in ("x", "y", "z"))
        except ValueError as error:
            raise ValueError(f"PLY lacks x/y/z vertex properties: {path}") from error
        rows = []
        for _ in range(vertex_count):
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY vertex data is truncated: {path}")
            values = line.split()
            rows.append([float(values[index]) for index in xyz])
    return _points(rows).astype(np.float32)


def _run(
    command: tuple[str, ...], *, timeout_seconds: float, tool_name: str
) -> tuple[str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env={**os.environ, "SEED": "0"},
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(
            f"{tool_name} failed with status {completed.returncode}: "
            f"{' '.join(command)}\n"
            f"{output[-4000:]}"
        )
    return output, elapsed


def run_tmc3(
    executable: Path,
    points: ArrayLike,
    *,
    rate_point: Tmc3RatePoint,
    work_dir: Path,
    position_bits: int = 12,
    timeout_seconds: float = 120.0,
) -> GpccResult:
    """Encode/decode one normalized cloud and return its actual TMC13 stream."""

    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"TMC13 executable is missing or not executable: {executable}"
        )
    if not 2 <= position_bits <= 24:
        raise ValueError("position_bits must be between 2 and 24")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    normalized = _points(points)
    if np.any(normalized < -1.0) or np.any(normalized > 1.0):
        raise ValueError("G-PCC input must lie in the declared [-1, 1] domain")
    levels = (1 << position_bits) - 1
    integer_points = np.rint((normalized + 1.0) * 0.5 * levels)

    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.ply"
    stream_path = work_dir / "stream.bin"
    encoder_reconstruction = work_dir / "encoder_reconstruction.ply"
    decoder_reconstruction = work_dir / "decoder_reconstruction.ply"
    write_ascii_ply(input_path, integer_points)

    executable_string = str(executable.resolve())
    encoder_command = (
        executable_string,
        "--mode=0",
        "--frameCount=1",
        f"--uncompressedDataPath={input_path}",
        f"--compressedStreamPath={stream_path}",
        f"--reconstructedDataPath={encoder_reconstruction}",
        "--outputBinaryPly=0",
        "--disableAttributeCoding=1",
        "--mergeDuplicatedPoints=1",
        *rate_point.encoder_args,
    )
    encoder_stdout, encode_seconds = _run(
        encoder_command, timeout_seconds=timeout_seconds, tool_name="TMC13"
    )
    if not stream_path.is_file() or not stream_path.stat().st_size:
        raise RuntimeError("TMC13 produced no compressed stream")

    decoder_command = (
        executable_string,
        "--mode=1",
        f"--compressedStreamPath={stream_path}",
        f"--reconstructedDataPath={decoder_reconstruction}",
        "--outputBinaryPly=0",
    )
    decoder_stdout, decode_seconds = _run(
        decoder_command, timeout_seconds=timeout_seconds, tool_name="TMC13"
    )
    decoded_integer = read_ascii_ply(decoder_reconstruction).astype(np.float64)
    reconstruction = (decoded_integer * (2.0 / levels) - 1.0).astype(np.float32)
    return GpccResult(
        reconstruction=reconstruction,
        stream_bytes=stream_path.stat().st_size,
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        encoder_command=encoder_command,
        decoder_command=decoder_command,
        encoder_stdout=encoder_stdout,
        decoder_stdout=decoder_stdout,
    )


def run_pc_error(
    executable: Path,
    original: ArrayLike,
    reconstruction: ArrayLike,
    normals: ArrayLike,
    *,
    work_dir: Path,
    position_bits: int = 12,
    timeout_seconds: float = 120.0,
) -> OfficialMetricResult:
    """Run MPEG ``pc_error`` on the declared integer coordinate grid."""

    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"pc_error executable is missing or not executable: {executable}"
        )
    if not 2 <= position_bits <= 24:
        raise ValueError("position_bits must be between 2 and 24")
    original_array = _points(original)
    reconstruction_array = _points(reconstruction)
    normal_array = _points(normals)
    if original_array.shape != normal_array.shape:
        raise ValueError("original points and normals must have matching shapes")
    if np.any(original_array < -1.0) or np.any(original_array > 1.0):
        raise ValueError("pc_error original must lie in the declared [-1, 1] domain")

    levels = (1 << position_bits) - 1

    def quantize(array: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.rint((array + 1.0) * 0.5 * levels)

    work_dir.mkdir(parents=True, exist_ok=True)
    original_path = work_dir / "original.ply"
    reconstruction_path = work_dir / "reconstruction.ply"
    normals_path = work_dir / "original_normals.ply"
    write_ascii_ply(original_path, quantize(original_array))
    write_ascii_ply(reconstruction_path, quantize(reconstruction_array))
    write_ascii_ply_with_normals(normals_path, quantize(original_array), normal_array)
    command = (
        str(executable.resolve()),
        f"--fileA={original_path}",
        f"--fileB={reconstruction_path}",
        f"--inputNorm={normals_path}",
        "--hausdorff=1",
        f"--resolution={levels}",
        "--dropdups=2",
    )
    stdout, elapsed = _run(
        command, timeout_seconds=timeout_seconds, tool_name="pc_error"
    )
    number = r"([-+0-9.eE]+|[-+]?inf|nan)"
    patterns = {
        "d1_mse": rf"mseF\s+\(p2point\):\s*{number}",
        "d1_psnr_db": rf"mseF,PSNR\s+\(p2point\):\s*{number}",
        "d2_mse": rf"mseF\s+\(p2plane\):\s*{number}",
        "d2_psnr_db": rf"mseF,PSNR\s+\(p2plane\):\s*{number}",
        "d1_hausdorff": rf"h\.\s+\(p2point\):\s*{number}",
        "d1_hausdorff_psnr_db": (rf"h\.,PSNR\s+\(p2point\):\s*{number}"),
        "d2_hausdorff": rf"h\.\s+\(p2plane\):\s*{number}",
        "d2_hausdorff_psnr_db": (rf"h\.,PSNR\s+\(p2plane\):\s*{number}"),
    }
    metrics = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match is None:
            raise RuntimeError(f"pc_error output is missing {name}\n{stdout[-4000:]}")
        metrics[name] = float(match.group(1))
    return OfficialMetricResult(metrics, elapsed, command, stdout)
