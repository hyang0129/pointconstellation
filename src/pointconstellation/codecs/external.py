"""Auditable adapter for third-party learned point-cloud codecs.

The adapter deliberately treats an external codec as a complete black-box
compress/decompress program.  It never imports third-party Python packages into
the Point Constellation environment, and it measures the actual emitted stream.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.codecs.gpcc import read_ascii_ply, write_ascii_ply

PLACEHOLDERS = frozenset(
    {
        "input",
        "stream",
        "reconstruction",
        "work_dir",
        "upstream_dir",
        "checkpoint_dir",
        "inputs",
        "streams",
        "reconstructions",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ply_xyz(path: Path, *, allow_empty: bool = False) -> NDArray[np.float32]:
    """Read x/y/z from ASCII or scalar-property binary PLY vertex data."""

    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        format_name = None
        vertex_count = None
        vertex_properties: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY header is unterminated: {path}")
            fields = raw.decode("ascii").strip().split()
            if fields[:1] == ["format"]:
                format_name = fields[1]
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields[:1] == ["element"]:
                in_vertex = False
            elif fields[:1] == ["property"] and in_vertex:
                if fields[1] == "list":
                    raise ValueError("list-valued vertex properties are unsupported")
                vertex_properties.append((fields[1], fields[2]))
            elif fields[:1] == ["end_header"]:
                data_offset = handle.tell()
                break
    if vertex_count == 0 and allow_empty:
        return np.empty((0, 3), dtype=np.float32)
    if format_name == "ascii":
        return read_ascii_ply(path)
    if format_name not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format {format_name}: {path}")
    if vertex_count is None or vertex_count < 1:
        raise ValueError(f"PLY has no vertices: {path}")
    names = [name for _, name in vertex_properties]
    if not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError(f"PLY lacks x/y/z vertex properties: {path}")
    type_codes = {
        "char": "b",
        "int8": "b",
        "uchar": "B",
        "uint8": "B",
        "short": "h",
        "int16": "h",
        "ushort": "H",
        "uint16": "H",
        "int": "i",
        "int32": "i",
        "uint": "I",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    try:
        codes = "".join(type_codes[data_type] for data_type, _ in vertex_properties)
    except KeyError as error:
        raise ValueError(f"unsupported PLY scalar type: {error.args[0]}") from error
    endian = "<" if format_name == "binary_little_endian" else ">"
    row_struct = struct.Struct(endian + codes)
    xyz_indices = tuple(names.index(axis) for axis in ("x", "y", "z"))
    points = np.empty((vertex_count, 3), dtype=np.float32)
    with path.open("rb") as handle:
        handle.seek(data_offset)
        for index in range(vertex_count):
            row = handle.read(row_struct.size)
            if len(row) != row_struct.size:
                raise ValueError(f"PLY vertex data is truncated: {path}")
            values = row_struct.unpack(row)
            points[index] = [values[axis] for axis in xyz_indices]
    return points


@dataclass(frozen=True)
class ExternalCodecSpec:
    """Pinned command and coordinate contract for one external codec point."""

    name: str
    upstream_url: str
    upstream_commit: str
    upstream_dir: str
    checkpoint_dir: str
    compress_command: tuple[str, ...]
    decompress_command: tuple[str, ...] = ()
    position_bits: int = 12
    timeout_seconds: float = 600.0
    model_bytes: int | None = None
    environment_manifest: str | None = None
    environment_variables: tuple[tuple[str, str], ...] = ()
    checkout_diff_sha256: str | None = None
    allow_empty_reconstruction: bool = False

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("external codec name must be nonempty without whitespace")
        if not self.upstream_url.startswith("https://"):
            raise ValueError("external codec upstream_url must use HTTPS")
        if len(self.upstream_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.upstream_commit
        ):
            raise ValueError("upstream_commit must be a lowercase 40-character SHA")
        if not self.compress_command:
            raise ValueError("compress_command cannot be empty")
        if not 2 <= self.position_bits <= 24:
            raise ValueError("position_bits must be between 2 and 24")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.model_bytes is not None and self.model_bytes < 0:
            raise ValueError("model_bytes cannot be negative")
        if self.checkout_diff_sha256 is not None and (
            len(self.checkout_diff_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.checkout_diff_sha256
            )
        ):
            raise ValueError("checkout_diff_sha256 must be a lowercase SHA-256")
        names = [name for name, _ in self.environment_variables]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("environment variable names must be nonempty and unique")
        for command in (self.compress_command, self.decompress_command):
            for argument in command:
                fields = {
                    field.split("}", 1)[0]
                    for field in argument.split("{")[1:]
                    if "}" in field
                }
                unknown = fields - PLACEHOLDERS
                if unknown:
                    raise ValueError(f"unknown command placeholders: {sorted(unknown)}")

    @classmethod
    def from_json(cls, path: Path) -> ExternalCodecSpec:
        values = json.loads(path.read_text())
        for key in ("compress_command", "decompress_command"):
            if key in values:
                values[key] = tuple(values[key])
        if "environment_variables" in values:
            values["environment_variables"] = tuple(
                tuple(item) for item in values["environment_variables"]
            )
        return cls(**values)


@dataclass(frozen=True)
class ExternalCodecResult:
    """Measured output of one complete external-codec round trip."""

    reconstruction: NDArray[np.float32]
    stream_bytes: int
    stream_sha256: str
    reconstruction_sha256: str
    encode_seconds: float
    decode_seconds: float
    compress_command: tuple[str, ...]
    decompress_command: tuple[str, ...]
    compress_output: str
    decompress_output: str
    upstream_commit: str
    checkout_diff_sha256: str
    environment: dict[str, Any] | None


def _commit(path: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(
            f"cannot resolve external codec commit in {path}: "
            f"{(completed.stdout + completed.stderr)[-2000:]}"
        )
    return completed.stdout.strip()


def _checkout_diff_sha256(path: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(path), "diff", "--binary", "HEAD"),
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(
            f"cannot resolve external codec patch in {path}: "
            f"{completed.stderr[-2000:].decode(errors='replace')}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _run(
    command: tuple[str, ...],
    *,
    timeout: float,
    stage: str,
    environment_variables: tuple[tuple[str, str], ...],
) -> tuple[str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "PYTHONHASHSEED": "0",
            **dict(environment_variables),
        },
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(
            f"external codec {stage} failed with status {completed.returncode}: "
            f"{' '.join(command)}\n{output[-4000:]}"
        )
    return output, elapsed


def _render(
    command: tuple[str, ...],
    *,
    input_path: Path,
    stream_path: Path,
    reconstruction_path: Path,
    work_dir: Path,
    upstream_dir: Path,
    checkpoint_dir: Path,
) -> tuple[str, ...]:
    values = {
        "input": str(input_path.resolve()),
        "stream": str(stream_path.resolve()),
        "reconstruction": str(reconstruction_path.resolve()),
        "work_dir": str(work_dir.resolve()),
        "upstream_dir": str(upstream_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
    }
    return tuple(argument.format_map(values) for argument in command)


def _render_batch(
    command: tuple[str, ...],
    *,
    input_paths: tuple[Path, ...],
    stream_paths: tuple[Path, ...],
    reconstruction_paths: tuple[Path, ...],
    work_dir: Path,
    upstream_dir: Path,
    checkpoint_dir: Path,
) -> tuple[str, ...]:
    expansions = {
        "{inputs}": tuple(str(path.resolve()) for path in input_paths),
        "{streams}": tuple(str(path.resolve()) for path in stream_paths),
        "{reconstructions}": tuple(
            str(path.resolve()) for path in reconstruction_paths
        ),
    }
    values = {
        "work_dir": str(work_dir.resolve()),
        "upstream_dir": str(upstream_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
    }
    rendered: list[str] = []
    for argument in command:
        if argument in expansions:
            rendered.extend(expansions[argument])
        else:
            if any(token in argument for token in expansions):
                raise ValueError("plural path placeholders must be whole arguments")
            rendered.append(argument.format_map(values))
    return tuple(rendered)


def _validate_checkout(
    spec: ExternalCodecSpec,
) -> tuple[Path, Path, str, str, dict[str, Any] | None]:
    upstream_dir = Path(spec.upstream_dir)
    checkpoint_dir = Path(spec.checkpoint_dir)
    if not upstream_dir.is_dir():
        raise FileNotFoundError(
            f"external upstream checkout is missing: {upstream_dir}"
        )
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"external checkpoint path is missing: {checkpoint_dir}"
        )
    actual_commit = _commit(upstream_dir)
    if actual_commit != spec.upstream_commit:
        raise RuntimeError(
            f"external codec commit mismatch: expected {spec.upstream_commit}, "
            f"found {actual_commit}"
        )
    actual_diff = _checkout_diff_sha256(upstream_dir)
    expected_diff = spec.checkout_diff_sha256 or hashlib.sha256(b"").hexdigest()
    if actual_diff != expected_diff:
        raise RuntimeError(
            f"external codec checkout patch mismatch: expected {expected_diff}, "
            f"found {actual_diff}"
        )
    environment = None
    if spec.environment_manifest is not None:
        manifest_path = Path(spec.environment_manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"external environment manifest is missing: {manifest_path}"
            )
        environment = json.loads(manifest_path.read_text())
    return upstream_dir, checkpoint_dir, actual_commit, actual_diff, environment


def _validate_points(points: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("points must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all() or np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("external codec input must be finite and lie in [-1, 1]")
    return array


def run_external_codec(
    spec: ExternalCodecSpec,
    points: ArrayLike,
    *,
    work_dir: Path,
) -> ExternalCodecResult:
    """Run a pinned external codec and return its measured decoded geometry."""

    upstream_dir, checkpoint_dir, actual_commit, actual_diff, environment = (
        _validate_checkout(spec)
    )
    array = _validate_points(points)

    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.ply"
    stream_path = work_dir / "stream.bin"
    reconstruction_path = work_dir / "reconstruction.ply"
    if stream_path.exists() or reconstruction_path.exists():
        raise FileExistsError(
            "external codec work directory already contains output files; "
            "use a fresh per-run directory"
        )
    levels = (1 << spec.position_bits) - 1
    integer_points = np.rint((array + 1.0) * 0.5 * levels)
    write_ascii_ply(input_path, integer_points)
    render = dict(
        input_path=input_path,
        stream_path=stream_path,
        reconstruction_path=reconstruction_path,
        work_dir=work_dir,
        upstream_dir=upstream_dir,
        checkpoint_dir=checkpoint_dir,
    )
    compress_command = _render(spec.compress_command, **render)
    compress_output, encode_seconds = _run(
        compress_command,
        timeout=spec.timeout_seconds,
        stage="compression",
        environment_variables=spec.environment_variables,
    )
    decompress_command = _render(spec.decompress_command, **render)
    if decompress_command:
        decompress_output, decode_seconds = _run(
            decompress_command,
            timeout=spec.timeout_seconds,
            stage="decompression",
            environment_variables=spec.environment_variables,
        )
    else:
        decompress_output = ""
        decode_seconds = 0.0
    if not stream_path.is_file() or stream_path.stat().st_size < 1:
        raise RuntimeError("external codec produced no nonempty stream")
    if not reconstruction_path.is_file():
        raise RuntimeError("external codec produced no decoded PLY")
    decoded_integer = _read_ply_xyz(
        reconstruction_path, allow_empty=spec.allow_empty_reconstruction
    ).astype(np.float64)
    if len(decoded_integer) and (
        np.any(decoded_integer < 0) or np.any(decoded_integer > levels)
    ):
        raise RuntimeError("external decoded coordinates lie outside the declared grid")
    reconstruction = (decoded_integer * (2.0 / levels) - 1.0).astype(np.float32)
    return ExternalCodecResult(
        reconstruction=reconstruction,
        stream_bytes=stream_path.stat().st_size,
        stream_sha256=_sha256(stream_path),
        reconstruction_sha256=_sha256(reconstruction_path),
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        compress_command=compress_command,
        decompress_command=decompress_command,
        compress_output=compress_output,
        decompress_output=decompress_output,
        upstream_commit=actual_commit,
        checkout_diff_sha256=actual_diff,
        environment=environment,
    )


def run_external_codec_batch(
    spec: ExternalCodecSpec,
    point_clouds: tuple[ArrayLike, ...],
    *,
    work_dirs: tuple[Path, ...],
) -> tuple[ExternalCodecResult, ...]:
    """Run several clouds in one pinned process to amortize model startup."""

    if not point_clouds or len(point_clouds) != len(work_dirs):
        raise ValueError("point_clouds and work_dirs must have the same nonzero length")
    if len(set(work_dirs)) != len(work_dirs):
        raise ValueError("batch work directories must be unique")
    required_compress = {"{inputs}", "{streams}"}
    reconstruction_from_compress = "{reconstructions}" in spec.compress_command
    reconstruction_from_decompress = {"{streams}", "{reconstructions}"}.issubset(
        spec.decompress_command
    )
    if not required_compress.issubset(spec.compress_command) or not (
        reconstruction_from_compress or reconstruction_from_decompress
    ):
        raise ValueError(
            "batch commands require input and stream expansion plus a decoded "
            "reconstruction output"
        )
    upstream_dir, checkpoint_dir, actual_commit, actual_diff, environment = (
        _validate_checkout(spec)
    )
    arrays = tuple(_validate_points(points) for points in point_clouds)
    input_paths = tuple(path / "input.ply" for path in work_dirs)
    stream_paths = tuple(path / "stream.bin" for path in work_dirs)
    reconstruction_paths = tuple(path / "reconstruction.ply" for path in work_dirs)
    levels = (1 << spec.position_bits) - 1
    for path, input_path, stream_path, reconstruction_path, array in zip(
        work_dirs,
        input_paths,
        stream_paths,
        reconstruction_paths,
        arrays,
        strict=True,
    ):
        path.mkdir(parents=True, exist_ok=True)
        if stream_path.exists() or reconstruction_path.exists():
            raise FileExistsError(
                "external codec work directory already contains output files; "
                "use fresh per-run directories"
            )
        integer_points = np.rint((array + 1.0) * 0.5 * levels)
        write_ascii_ply(input_path, integer_points)
    common_work_dir = Path(
        os.path.commonpath([str(path.resolve()) for path in work_dirs])
    )
    render = dict(
        input_paths=input_paths,
        stream_paths=stream_paths,
        reconstruction_paths=reconstruction_paths,
        work_dir=common_work_dir,
        upstream_dir=upstream_dir,
        checkpoint_dir=checkpoint_dir,
    )
    compress_command = _render_batch(spec.compress_command, **render)
    compress_output, encode_seconds = _run(
        compress_command,
        timeout=spec.timeout_seconds,
        stage="batch compression",
        environment_variables=spec.environment_variables,
    )
    decompress_command = _render_batch(spec.decompress_command, **render)
    if decompress_command:
        decompress_output, decode_seconds = _run(
            decompress_command,
            timeout=spec.timeout_seconds,
            stage="batch decompression",
            environment_variables=spec.environment_variables,
        )
    else:
        decompress_output = ""
        decode_seconds = 0.0
    results = []
    count = len(arrays)
    for stream_path, reconstruction_path in zip(
        stream_paths, reconstruction_paths, strict=True
    ):
        if not stream_path.is_file() or stream_path.stat().st_size < 1:
            raise RuntimeError("external codec produced no nonempty stream")
        if not reconstruction_path.is_file():
            raise RuntimeError("external codec produced no decoded PLY")
        decoded_integer = _read_ply_xyz(
            reconstruction_path, allow_empty=spec.allow_empty_reconstruction
        ).astype(np.float64)
        if len(decoded_integer) and (
            np.any(decoded_integer < 0) or np.any(decoded_integer > levels)
        ):
            raise RuntimeError(
                "external decoded coordinates lie outside the declared grid"
            )
        reconstruction = (decoded_integer * (2.0 / levels) - 1.0).astype(np.float32)
        results.append(
            ExternalCodecResult(
                reconstruction=reconstruction,
                stream_bytes=stream_path.stat().st_size,
                stream_sha256=_sha256(stream_path),
                reconstruction_sha256=_sha256(reconstruction_path),
                encode_seconds=encode_seconds / count,
                decode_seconds=decode_seconds / count,
                compress_command=compress_command,
                decompress_command=decompress_command,
                compress_output=compress_output,
                decompress_output=decompress_output,
                upstream_commit=actual_commit,
                checkout_diff_sha256=actual_diff,
                environment=environment,
            )
        )
    return tuple(results)
