from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pointconstellation.data import file_sha256
from pointconstellation.retrained_codec_benchmark import (
    RetrainedCodecBenchmarkConfig,
    _checked_array,
)


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
