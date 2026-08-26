from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from pointconstellation.codecs.draco import DracoCodecSpec, run_draco
from pointconstellation.draco_benchmark import (
    DracoBenchmarkConfig,
    default_draco_arms,
    evaluate_draco_arm,
    fps_subset,
)


def _executable(path: Path, source: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + source)
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _fake_draco(tmp_path: Path) -> tuple[Path, Path]:
    encoder = _executable(
        tmp_path / "draco_encoder",
        """from pathlib import Path
import sys
assert '-point_cloud' in sys.argv
values = sys.argv[1:]
values.remove('-point_cloud')
arguments = dict(zip(values[::2], values[1::2]))
Path(arguments['-o']).write_bytes(Path(arguments['-i']).read_bytes())
""",
    )
    decoder = _executable(
        tmp_path / "draco_decoder",
        """from pathlib import Path
import sys
arguments = dict(zip(sys.argv[1::2], sys.argv[2::2]))
Path(arguments['-o']).write_bytes(Path(arguments['-i']).read_bytes())
""",
    )
    return encoder, decoder


def _fake_pc_error(path: Path) -> Path:
    return _executable(
        path,
        """print('mseF      (p2point): 1.25')
print('mseF,PSNR (p2point): 42.5')
print('mseF      (p2plane): 0.75')
print('mseF,PSNR (p2plane): 45.0')
print('h.        (p2point): 3.0')
print('h.,PSNR   (p2point): 35.0')
print('h.        (p2plane): 2.0')
print('h.,PSNR   (p2plane): 38.0')
""",
    )


def test_draco_adapter_independently_decodes_and_counts_payload(tmp_path: Path) -> None:
    encoder, decoder = _fake_draco(tmp_path)
    spec = DracoCodecSpec(
        encoder_executable=str(encoder),
        decoder_executable=str(decoder),
        encoder_sha256=hashlib.sha256(encoder.read_bytes()).hexdigest(),
        decoder_sha256=hashlib.sha256(decoder.read_bytes()).hexdigest(),
    )
    points = np.asarray([[-1.0, -0.5, 0.0], [0.25, 0.5, 1.0]], dtype=np.float32)

    result = run_draco(spec, points, quantization_bits=10, work_dir=tmp_path / "run")

    stream = tmp_path / "run/stream.drc"
    assert result.payload_bytes == result.stream_bytes == stream.stat().st_size
    assert result.encoder_command[1] == "-point_cloud"
    assert result.decoder_command[0] != result.encoder_command[0]
    assert np.array_equal(result.reconstruction, points)


@pytest.mark.skipif(
    shutil.which("draco_encoder") is None or shutil.which("draco_decoder") is None,
    reason="Draco command-line binaries are not installed",
)
def test_installed_draco_round_trip_on_coordinate_fixture(tmp_path: Path) -> None:
    encoder = shutil.which("draco_encoder")
    decoder = shutil.which("draco_decoder")
    assert encoder is not None and decoder is not None
    points = np.asarray(
        [[-1.0, -1.0, -1.0], [-0.25, 0.5, 0.75], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )

    result = run_draco(
        DracoCodecSpec(encoder, decoder),
        points,
        quantization_bits=12,
        work_dir=tmp_path / "installed",
    )

    assert result.payload_bytes > 0
    assert result.reconstruction.shape == points.shape
    assert np.isfinite(result.reconstruction).all()


def test_draco_arm_list_and_fps_are_deterministic() -> None:
    arms = default_draco_arms(8)
    points = np.arange(48, dtype=np.float32).reshape(16, 3) / 48

    first = fps_subset(points, 8)
    second = fps_subset(points, 8)

    assert len(arms) == 16
    assert {
        arm.quantization_bits for arm in arms if arm.input_kind != "full_cloud"
    } == {
        8,
        10,
        12,
    }
    assert {
        arm.quantization_bits for arm in arms if arm.input_kind == "full_cloud"
    } == {
        3,
        4,
        5,
        6,
    }
    assert np.array_equal(first, second)
    assert len(np.unique(first, axis=0)) == 8


def test_draco_benchmark_row_has_registry_rate_and_model_fields(
    tmp_path: Path,
) -> None:
    encoder, decoder = _fake_draco(tmp_path)
    metric = _fake_pc_error(tmp_path / "pc_error")
    arms = default_draco_arms(2)
    arm = next(
        candidate
        for candidate in arms
        if candidate.input_kind == "fps"
        and candidate.quantization_bits == 8
        and candidate.reconstruction_route == "frozen_decoder"
    )
    config = DracoBenchmarkConfig(
        codec=DracoCodecSpec(str(encoder), str(decoder)),
        arms=arms,
        pc_error_executable=str(metric),
    )
    source = np.asarray(
        [[-1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1))

    row = evaluate_draco_arm(
        config,
        arm,
        source,
        normals,
        work_dir=tmp_path / "benchmark",
        split="validation",
        family="chair",
        model_id="chair_0001",
        sample_id=1,
        frozen_decoder=lambda coordinates: np.repeat(coordinates, 2, axis=0),
        frozen_decoder_model_bytes=1234,
    )

    assert row["family"] == "chair"
    assert row["codec_family"] == "draco"
    assert row["payload_bytes"] == row["stream_bytes"] > 0
    assert row["model_bytes"] == 1234
    assert row["d1_mse"] == pytest.approx(1.25)
