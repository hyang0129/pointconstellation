from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pointconstellation.external_codec_training import (
    ExactExternalRetrainConfig,
    ExternalTrainingArm,
    _native_blocks,
    _quantize_source,
    _safe_extract,
)


def _arm(*, bits: int = 6, level: int = 0) -> ExternalTrainingArm:
    return ExternalTrainingArm(
        name="low_rate",
        position_bits=bits,
        octree_level=level,
        lambdas=("1.00e-06",),
        max_steps=500,
    )


def test_exact_external_training_quantizes_only_coordinates() -> None:
    points = np.asarray(
        [[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
        dtype=np.float32,
    )

    quantized = _quantize_source(points, 6)
    blocks = _native_blocks(quantized, _arm())

    assert quantized.shape == (2, 3)
    assert quantized.min() == 0
    assert quantized.max() == 63
    assert len(blocks) == 1
    assert blocks[0].shape == (2, 3)


def test_native_q9_partition_produces_only_64_cubed_blocks() -> None:
    arm = _arm(bits=9, level=3)
    points = np.asarray([[0, 0, 0], [63, 63, 63], [64, 64, 64], [511, 511, 511]])

    blocks = _native_blocks(points, arm)

    assert len(blocks) == 3
    assert all(block.min() >= 0 and block.max() < 64 for block in blocks)


def test_external_training_config_rejects_non_native_block_shape() -> None:
    with pytest.raises(ValueError, match="native 64"):
        _arm(bits=9, level=2)


def test_external_training_config_pins_exact_split() -> None:
    config = ExactExternalRetrainConfig.from_json(
        Path("configs/experiment_020_external_retrain.json")
    )

    assert config.expected_stability_manifest_sha256 == (
        "d44014dd2313b8815562cde9df2ba1927e1110fcfbc218428a7db39ef6b829ac"
    )
    assert config.arm("native_q9_oct3").octree_level == 3
    assert config.arm("low_rate_q6_global").octree_level == 0
    assert len(config.arm("low_rate_q6_global").lambdas) == 5
    assert config.checkout_diff_sha256 == (
        "419ac2ca018fd7402a5624c075b9b660d1af7ce2ee2bdd5bc66edb0241e11666"
    )


def test_safe_extract_rejects_link_members(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    payload = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=payload, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as tar,
    ):
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escape"
        tar.addfile(info)
    archive.write_bytes(payload.getvalue())

    with pytest.raises(ValueError, match="unsafe"):
        _safe_extract(archive, tmp_path / "output")
