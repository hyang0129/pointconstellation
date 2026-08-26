from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pointconstellation.data import file_sha256
from pointconstellation.retrained_codec_benchmark import (
    RetrainedCodecBenchmarkConfig,
    _checked_array,
    _gzip_stream_breakdown,
    _rmse_or_none,
    _valid_only_rmse,
)


def test_external_gzip_header_and_payload_sum_to_stream_size(tmp_path: Path) -> None:
    path = tmp_path / "stream.bin"
    path.write_bytes(gzip.compress(b"learned geometry payload", mtime=0))

    breakdown = _gzip_stream_breakdown(path)

    assert breakdown.header_bytes + breakdown.payload_bytes == path.stat().st_size
    assert breakdown.header_bytes == 18
    assert breakdown.payload_bytes > 0


def test_retrained_rate_config_requires_sealed_evaluation_manifest() -> None:
    with pytest.raises(ValueError, match="pinned"):
        RetrainedCodecBenchmarkConfig(
            codec_manifest="codec.json",
            evaluation_root="evaluation",
            expected_evaluation_manifest_sha256="short",
            pc_error_executable="pc_error",
            output_dir="output",
        )


def test_checked_evaluation_array_rejects_hash_drift(tmp_path: Path) -> None:
    path = tmp_path / "source.npy"
    np.save(path, np.zeros((8, 3), dtype=np.float32), allow_pickle=False)

    loaded = _checked_array(tmp_path, "source.npy", file_sha256(path))

    assert loaded.shape == (8, 3)
    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        _checked_array(tmp_path, "source.npy", "0" * 64)


def test_failed_cloud_invalidates_rate_point_aggregate() -> None:
    values = [1.0, None, 9.0]

    assert _rmse_or_none(values) is None
    assert _valid_only_rmse(values) == pytest.approx(np.sqrt(5.0))


def test_rmse_helpers_handle_fully_valid_and_fully_failed_rows() -> None:
    assert _rmse_or_none([1.0, 9.0]) == pytest.approx(np.sqrt(5.0))
    assert _rmse_or_none([]) is None
    assert _valid_only_rmse([None, None]) is None
