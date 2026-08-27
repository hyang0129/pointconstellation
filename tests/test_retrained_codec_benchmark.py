from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pointconstellation.data import file_sha256
from pointconstellation.published_codec_benchmark import (
    _diversity_contract_summary,
)
from pointconstellation.retrained_codec_benchmark import (
    RetrainedCodecBenchmarkConfig,
    _checked_array,
    _rmse_or_none,
    _valid_only_rmse,
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


def test_retrained_rate_config_validates_diversity_fraction() -> None:
    values = {
        "codec_manifest": "codec.json",
        "evaluation_root": "evaluation",
        "expected_evaluation_manifest_sha256": "0" * 64,
        "pc_error_executable": "pc_error",
        "output_dir": "output",
    }

    assert (
        RetrainedCodecBenchmarkConfig(**values).minimum_unique_reconstruction_fraction
        == 0.9
    )
    with pytest.raises(ValueError, match="minimum_unique_reconstruction_fraction"):
        RetrainedCodecBenchmarkConfig(
            **values, minimum_unique_reconstruction_fraction=1.1
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


def test_retrained_diversity_contract_detects_near_constant_output() -> None:
    rows = [
        {
            "stream_sha256": f"{index % 13:064x}",
            "reconstruction_sha256": f"{int(index == 31):064x}",
        }
        for index in range(32)
    ]

    summary = _diversity_contract_summary(
        rows, minimum_unique_reconstruction_fraction=0.9
    )

    assert summary == {
        "unique_streams": 13,
        "unique_reconstructions": 2,
        "constant_output": True,
        "rate_point_valid": False,
    }
