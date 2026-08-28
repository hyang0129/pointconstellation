"""Pinned PCGCv2 specifications and complete-payload stream framing."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pointconstellation.codecs.external import ExternalCodecSpec

PCGCV2_UPSTREAM_URL = "https://github.com/NJUVISION/PCGCv2.git"
PCGCV2_UPSTREAM_COMMIT = "88ff2a18b1b3cac89eef66997cc4e8bcf4fb0420"
PCGCV2_PAYLOAD_SUFFIXES = (
    "_C.bin",
    "_F.bin",
    "_H.bin",
    "_num_points.bin",
)
_STREAM_MAGIC = b"PCGCV2\x00\x01"
_LENGTH = struct.Struct("<Q")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_set_sha256(points: ArrayLike) -> str:
    """Hash decoded coordinates independently of their serialized order."""

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("decoded point set must be a finite N x 3 array")
    if len(array):
        order = np.lexsort((array[:, 2], array[:, 1], array[:, 0]))
        array = np.ascontiguousarray(array[order])
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(array)))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _hash(value: str | None, *, label: str) -> None:
    if value is not None and (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def pack_pcgcv2_payloads(prefix: Path, stream_path: Path) -> int:
    """Frame every upstream payload and return the complete stream size."""

    if stream_path.exists():
        raise FileExistsError(f"PCGCv2 stream already exists: {stream_path}")
    payloads = []
    for suffix in PCGCV2_PAYLOAD_SUFFIXES:
        path = Path(str(prefix) + suffix)
        if not path.is_file():
            raise FileNotFoundError(f"PCGCv2 payload is missing: {path}")
        payloads.append(path.read_bytes())
    if any(not payload for payload in payloads):
        raise RuntimeError("PCGCv2 produced an empty component payload")
    with stream_path.open("xb") as handle:
        handle.write(_STREAM_MAGIC)
        for payload in payloads:
            handle.write(_LENGTH.pack(len(payload)))
            handle.write(payload)
    return stream_path.stat().st_size


def unpack_pcgcv2_payloads(stream_path: Path, prefix: Path) -> tuple[Path, ...]:
    """Validate a complete stream and restore the four upstream payloads."""

    payload = stream_path.read_bytes()
    if not payload.startswith(_STREAM_MAGIC):
        raise ValueError("PCGCv2 stream magic or version is invalid")
    offset = len(_STREAM_MAGIC)
    outputs = []
    for suffix in PCGCV2_PAYLOAD_SUFFIXES:
        if offset + _LENGTH.size > len(payload):
            raise ValueError("PCGCv2 stream is truncated before a payload length")
        size = _LENGTH.unpack_from(payload, offset)[0]
        offset += _LENGTH.size
        end = offset + size
        if size < 1 or end > len(payload):
            raise ValueError("PCGCv2 stream contains an invalid payload length")
        output = Path(str(prefix) + suffix)
        if output.exists():
            raise FileExistsError(f"PCGCv2 decoded payload already exists: {output}")
        output.write_bytes(payload[offset:end])
        outputs.append(output)
        offset = end
    if offset != len(payload):
        raise ValueError("PCGCv2 stream contains unaccounted trailing bytes")
    return tuple(outputs)


@dataclass(frozen=True)
class Pcgcv2RatePoint:
    """One released checkpoint and declared inference scaling point."""

    name: str
    checkpoint_subpath: str
    scaling_factor: float
    rho: float = 1.0
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError(
                "PCGCv2 rate-point name must be nonempty without whitespace"
            )
        if (
            Path(self.checkpoint_subpath).is_absolute()
            or ".." in Path(self.checkpoint_subpath).parts
        ):
            raise ValueError("PCGCv2 checkpoint_subpath must stay within the workspace")
        if self.scaling_factor <= 0 or self.rho <= 0:
            raise ValueError("PCGCv2 scaling_factor and rho must be positive")
        _hash(self.checkpoint_sha256, label="checkpoint_sha256")


@dataclass(frozen=True)
class Pcgcv2HarnessConfig:
    """ExternalCodecSpec factory for the PCGCv2 low-rate protocol."""

    workspace_root: str
    upstream_subpath: str
    environment_python_subpath: str
    environment_manifest_subpath: str
    adapter_script: str
    position_bits: tuple[int, ...]
    rate_points: tuple[Pcgcv2RatePoint, ...]
    upstream_url: str = PCGCV2_UPSTREAM_URL
    upstream_commit: str = PCGCV2_UPSTREAM_COMMIT
    checkout_diff_sha256: str | None = None
    timeout_seconds: float = 3600.0
    minimum_unique_reconstruction_fraction: float = 0.9

    def __post_init__(self) -> None:
        if self.upstream_url != PCGCV2_UPSTREAM_URL:
            raise ValueError("PCGCv2 upstream URL differs from the pinned repository")
        if self.upstream_commit != PCGCV2_UPSTREAM_COMMIT:
            raise ValueError("PCGCv2 upstream commit differs from the pinned checkout")
        if self.position_bits != (6, 7, 8):
            raise ValueError("PCGCv2 low-rate voxel grids must be exactly (6, 7, 8)")
        if not self.rate_points or len(
            {point.name for point in self.rate_points}
        ) != len(self.rate_points):
            raise ValueError("PCGCv2 rate points must be nonempty and unique")
        _hash(self.checkout_diff_sha256, label="checkout_diff_sha256")
        if self.timeout_seconds <= 0:
            raise ValueError("PCGCv2 timeout_seconds must be positive")
        if not 0 < self.minimum_unique_reconstruction_fraction <= 1:
            raise ValueError("minimum_unique_reconstruction_fraction must be in (0, 1]")

    @classmethod
    def from_json(cls, path: Path) -> Pcgcv2HarnessConfig:
        values = json.loads(path.read_text())
        values["position_bits"] = tuple(values["position_bits"])
        values["rate_points"] = tuple(
            Pcgcv2RatePoint(**point) for point in values["rate_points"]
        )
        return cls(**values)

    def external_spec(
        self,
        rate_point_name: str,
        position_bits: int,
        *,
        require_files: bool = True,
    ) -> ExternalCodecSpec:
        """Create a black-box spec whose stream counts all four payloads."""

        if position_bits not in self.position_bits:
            raise ValueError("undeclared PCGCv2 voxel precision")
        try:
            rate_point = next(
                point for point in self.rate_points if point.name == rate_point_name
            )
        except StopIteration as error:
            raise ValueError(f"unknown PCGCv2 rate point: {rate_point_name}") from error
        workspace = Path(self.workspace_root)
        python = workspace / self.environment_python_subpath
        checkpoint = workspace / rate_point.checkpoint_subpath
        adapter = Path(self.adapter_script)
        if require_files:
            for path, label in (
                (python, "environment Python"),
                (checkpoint, "checkpoint"),
                (adapter, "adapter script"),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"PCGCv2 {label} is missing: {path}")
            if rate_point.checkpoint_sha256 is not None and _sha256(checkpoint) != (
                rate_point.checkpoint_sha256
            ):
                raise RuntimeError("PCGCv2 checkpoint SHA-256 differs")
            model_bytes = checkpoint.stat().st_size
        else:
            model_bytes = checkpoint.stat().st_size if checkpoint.is_file() else None
        common = (
            str(python.resolve()),
            str(adapter.resolve()),
            "--upstream-dir",
            "{upstream_dir}",
            "--checkpoint",
            "{checkpoint_dir}",
            "--work-dir",
            "{work_dir}",
            "--position-bits",
            str(position_bits),
            "--scaling-factor",
            str(rate_point.scaling_factor),
            "--rho",
            str(rate_point.rho),
        )
        return ExternalCodecSpec(
            name=f"pcgcv2_{rate_point.name}_q{position_bits}",
            upstream_url=self.upstream_url,
            upstream_commit=self.upstream_commit,
            upstream_dir=str(workspace / self.upstream_subpath),
            checkpoint_dir=str(checkpoint),
            compress_command=(
                *common[:2],
                "encode",
                *common[2:],
                "--input",
                "{input}",
                "--stream",
                "{stream}",
            ),
            decompress_command=(
                *common[:2],
                "decode",
                *common[2:],
                "--stream",
                "{stream}",
                "--reconstruction",
                "{reconstruction}",
            ),
            position_bits=position_bits,
            timeout_seconds=self.timeout_seconds,
            model_bytes=model_bytes,
            environment_manifest=str(workspace / self.environment_manifest_subpath),
            checkout_diff_sha256=self.checkout_diff_sha256,
        )


def pcgcv2_diversity_summary(
    rows: list[dict[str, Any]],
    *,
    minimum_unique_reconstruction_fraction: float = 0.9,
) -> dict[str, Any]:
    """Reject empty, constant-stream, and near-constant-output rate points."""

    if not rows:
        raise ValueError("PCGCv2 diversity check requires at least one row")
    if not 0 < minimum_unique_reconstruction_fraction <= 1:
        raise ValueError("minimum_unique_reconstruction_fraction must be in (0, 1]")
    unique_streams = len({row["stream_sha256"] for row in rows})
    unique_reconstructions = len(
        {
            row.get("reconstruction_geometry_sha256", row["reconstruction_sha256"])
            for row in rows
        }
    )
    reconstruction_fraction = unique_reconstructions / len(rows)
    empty_reconstructions = sum(row.get("decoded_points", 1) == 0 for row in rows)
    constant_output = reconstruction_fraction < minimum_unique_reconstruction_fraction
    return {
        "clouds": len(rows),
        "unique_streams": unique_streams,
        "unique_reconstructions": unique_reconstructions,
        "unique_reconstruction_fraction": reconstruction_fraction,
        "empty_reconstructions": empty_reconstructions,
        "constant_output": constant_output,
        "rate_point_valid": (
            not empty_reconstructions and not constant_output and unique_streams > 1
        ),
    }
