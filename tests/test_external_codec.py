from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pointconstellation.codecs.external import (
    ExternalCodecSpec,
    _read_ply_xyz,
    run_external_codec,
    run_external_codec_batch,
)


def _upstream(tmp_path: Path) -> tuple[Path, str]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    script = upstream / "codec.py"
    script.write_text(
        """from pathlib import Path
import os
import shutil
import sys
if os.environ.get('CODEC_SENTINEL') != 'present':
    raise RuntimeError('adapter environment was not forwarded')
mode, input_path, stream_path, reconstruction_path = sys.argv[1:]
if mode == 'compress':
    payload = Path(input_path).read_bytes()
    Path(stream_path).write_bytes(payload[:31])
else:
    shutil.copyfile(input_path, reconstruction_path)
"""
    )
    subprocess.run(("git", "init", "-q", str(upstream)), check=True)
    subprocess.run(("git", "-C", str(upstream), "add", "codec.py"), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.name=Codec Test",
            "-c",
            "user.email=codec@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(upstream), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return upstream, commit


def _spec(tmp_path: Path, *, expected_commit: str | None = None) -> ExternalCodecSpec:
    upstream, commit = _upstream(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"python": sys.version.split()[0]}))
    return ExternalCodecSpec(
        name="fixture",
        upstream_url="https://example.invalid/fixture.git",
        upstream_commit=expected_commit or commit,
        upstream_dir=str(upstream),
        checkpoint_dir=str(checkpoint),
        compress_command=(
            sys.executable,
            "{upstream_dir}/codec.py",
            "compress",
            "{input}",
            "{stream}",
            "{reconstruction}",
        ),
        decompress_command=(
            sys.executable,
            "{upstream_dir}/codec.py",
            "decompress",
            "{input}",
            "{stream}",
            "{reconstruction}",
        ),
        position_bits=10,
        environment_manifest=str(environment),
        environment_variables=(("CODEC_SENTINEL", "present"),),
    )


def test_external_codec_requires_pinned_commit_and_counts_stream(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    points = np.asarray([[-1.0, -0.5, 0.0], [0.25, 0.5, 1.0]], dtype=np.float32)

    result = run_external_codec(spec, points, work_dir=tmp_path / "run")

    assert result.stream_bytes == 31
    assert result.upstream_commit == spec.upstream_commit
    assert len(result.stream_sha256) == 64
    assert result.environment == {"python": sys.version.split()[0]}
    assert np.allclose(result.reconstruction, points, atol=1.0 / 1023)


def test_external_codec_rejects_checkout_drift(tmp_path: Path) -> None:
    spec = _spec(tmp_path, expected_commit="0" * 40)
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="commit mismatch"):
        run_external_codec(spec, points, work_dir=tmp_path / "run")


def test_external_codec_refuses_to_overwrite_outputs(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    work_dir = tmp_path / "run"

    run_external_codec(spec, points, work_dir=work_dir)
    with pytest.raises(FileExistsError, match="fresh per-run directory"):
        run_external_codec(spec, points, work_dir=work_dir)


def test_external_codec_batches_clouds_in_one_process(tmp_path: Path) -> None:
    upstream, commit = _upstream(tmp_path)
    script = upstream / "batch_codec.py"
    script.write_text(
        """from pathlib import Path
import shutil
import sys
args = sys.argv[1:]
mode = args.pop(0)
if mode == 'compress':
    stream_index = args.index('--streams')
    inputs = args[:stream_index]
    streams = args[stream_index + 1:]
    for input_path, stream_path in zip(inputs, streams):
        Path(stream_path).write_bytes(Path(input_path).read_bytes())
else:
    reconstruction_index = args.index('--reconstructions')
    streams = args[:reconstruction_index]
    reconstructions = args[reconstruction_index + 1:]
    for stream_path, reconstruction_path in zip(streams, reconstructions):
        shutil.copyfile(stream_path, reconstruction_path)
"""
    )
    subprocess.run(("git", "-C", str(upstream), "add", "batch_codec.py"), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.name=Codec Test",
            "-c",
            "user.email=codec@example.invalid",
            "commit",
            "-qm",
            "batch fixture",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(upstream), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    spec = ExternalCodecSpec(
        name="batch_fixture",
        upstream_url="https://example.invalid/fixture.git",
        upstream_commit=commit,
        upstream_dir=str(upstream),
        checkpoint_dir=str(checkpoint),
        compress_command=(
            sys.executable,
            "{upstream_dir}/batch_codec.py",
            "compress",
            "{inputs}",
            "--streams",
            "{streams}",
        ),
        decompress_command=(
            sys.executable,
            "{upstream_dir}/batch_codec.py",
            "decompress",
            "{streams}",
            "--reconstructions",
            "{reconstructions}",
        ),
        position_bits=10,
    )
    point_clouds = (
        np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.25, -0.5, 0.75]], dtype=np.float32),
    )

    results = run_external_codec_batch(
        spec,
        point_clouds,
        work_dirs=(tmp_path / "run_0", tmp_path / "run_1"),
    )

    assert all(result.stream_bytes > 31 for result in results)
    assert all(result.decode_seconds > 0 for result in results)
    assert np.allclose(results[0].reconstruction, point_clouds[0], atol=1 / 1023)
    assert np.allclose(results[1].reconstruction, point_clouds[1], atol=1 / 1023)


def test_external_codec_spec_rejects_unknown_placeholders() -> None:
    with pytest.raises(ValueError, match="unknown command placeholders"):
        ExternalCodecSpec(
            name="bad",
            upstream_url="https://example.invalid/bad.git",
            upstream_commit="0" * 40,
            upstream_dir="upstream",
            checkpoint_dir="checkpoint",
            compress_command=("codec", "{target}"),
        )


def test_external_codec_spec_rejects_duplicate_environment_names() -> None:
    with pytest.raises(ValueError, match="nonempty and unique"):
        ExternalCodecSpec(
            name="bad_env",
            upstream_url="https://example.invalid/bad.git",
            upstream_commit="0" * 40,
            upstream_dir="upstream",
            checkpoint_dir="checkpoint",
            compress_command=("codec",),
            environment_variables=(("SAME", "one"), ("SAME", "two")),
        )


def test_external_codec_reads_binary_ply_with_extra_properties(tmp_path: Path) -> None:
    path = tmp_path / "binary.ply"
    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 2\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"end_header\n"
    )
    path.write_bytes(
        header
        + struct.pack("<fffB", 1.0, 2.0, 3.0, 255)
        + struct.pack("<fffB", 4.0, 5.0, 6.0, 0)
    )

    points = _read_ply_xyz(path)

    assert np.array_equal(points, np.asarray([[1, 2, 3], [4, 5, 6]]))
