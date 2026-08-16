from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pointconstellation.codecs.gpcc import (
    Tmc3RatePoint,
    read_ascii_ply,
    run_pc_error,
    run_tmc3,
    write_ascii_ply,
)


def _executable(path: Path, source: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + source)
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_ascii_ply_round_trip(tmp_path: Path) -> None:
    points = np.asarray([[-1.0, 0.25, 1.0], [0.5, -0.75, 0.0]], dtype=np.float32)
    path = tmp_path / "cloud.ply"

    write_ascii_ply(path, points)

    assert np.array_equal(read_ascii_ply(path), points)


def test_rate_point_rejects_managed_arguments() -> None:
    with pytest.raises(ValueError, match="managed option"):
        Tmc3RatePoint("bad", ("--mode=0",))
    with pytest.raises(ValueError, match="option=value"):
        Tmc3RatePoint("bad", ("--codingScale",))


def test_tmc3_adapter_counts_the_real_stream_and_round_trips(tmp_path: Path) -> None:
    executable = _executable(
        tmp_path / "fake_tmc3",
        """from pathlib import Path
import sys
options = dict(argument[2:].split('=', 1) for argument in sys.argv[1:])
stream = Path(options['compressedStreamPath'])
reconstruction = Path(options['reconstructedDataPath'])
if options['mode'] == '0':
    payload = Path(options['uncompressedDataPath']).read_bytes()
    stream.write_bytes(payload)
    reconstruction.write_bytes(payload)
else:
    reconstruction.write_bytes(stream.read_bytes())
""",
    )
    points = np.asarray([[-1.0, -0.5, 0.0], [0.25, 0.5, 1.0]], dtype=np.float32)

    result = run_tmc3(
        executable,
        points,
        rate_point=Tmc3RatePoint("lossless", ("--codingScale=1",)),
        work_dir=tmp_path / "run",
        position_bits=10,
    )

    assert result.stream_bytes == (tmp_path / "run/stream.bin").stat().st_size
    assert np.allclose(result.reconstruction, points, atol=1.0 / 1023)
    assert "--mode=0" in result.encoder_command
    assert "--mode=1" in result.decoder_command


def test_pc_error_adapter_parses_symmetric_d1_d2_metrics(tmp_path: Path) -> None:
    executable = _executable(
        tmp_path / "fake_pc_error",
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
    points = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    result = run_pc_error(
        executable,
        points,
        points,
        normals,
        work_dir=tmp_path / "metric",
        position_bits=10,
    )

    assert result.metrics["d1_mse"] == pytest.approx(1.25)
    assert result.metrics["d2_psnr_db"] == pytest.approx(45.0)
    assert result.metrics["d2_hausdorff_psnr_db"] == pytest.approx(38.0)
    assert os.path.samefile(result.command[0], executable)
