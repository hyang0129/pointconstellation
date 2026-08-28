from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from pointconstellation.published_codec_benchmark import (
    PccGeoCnnV2Manifest,
    PublishedCodecBenchmarkConfig,
    _rate_summaries,
    _trained_checkpoint_bytes,
)


def test_checked_pcc_geo_cnn_manifest_is_pinned_and_has_five_rates() -> None:
    manifest = PccGeoCnnV2Manifest.from_json(
        Path("configs/external/pcc_geo_cnn_v2_modelnet.json")
    )

    assert manifest.upstream_commit == ("b7a4ae2a548ad3c44a04af139dd77d804cf3a6fa")
    assert len(manifest.lambdas) == 5
    assert manifest.resolution == 1 << manifest.position_bits
    assert manifest.position_bits == 9
    assert manifest.octree_level == 3
    assert manifest.artifact_bytes == 5_523_313_425


def test_published_codec_config_rejects_test_alias() -> None:
    with pytest.raises(ValueError, match="validation and/or ood"):
        PublishedCodecBenchmarkConfig(
            codec_manifest="codec.json",
            stability_config="stability.json",
            pc_error_executable="pc_error",
            splits=("test",),
        )


def test_rate_summary_uses_actual_streams_and_aggregate_mse() -> None:
    rows = [
        {
            "lambda": "1.00e-04",
            "split": "validation",
            "stream_bytes": 40,
            "actual_stream_bpp": 0.15625,
            "actual_stream_bpov": 0.20,
            "chamfer_mse": 0.04,
            "d1_mse": 16.0,
            "d2_mse": 9.0,
            "d1_psnr_db": 40.0,
            "d2_psnr_db": 42.0,
            "encode_seconds": 1.0,
            "official_metric_seconds": 0.1,
            "model_bytes": 1000,
        },
        {
            "lambda": "1.00e-04",
            "split": "validation",
            "stream_bytes": 60,
            "actual_stream_bpp": 0.234375,
            "actual_stream_bpov": 0.30,
            "chamfer_mse": 0.09,
            "d1_mse": 36.0,
            "d2_mse": 25.0,
            "d1_psnr_db": 38.0,
            "d2_psnr_db": 40.0,
            "encode_seconds": 2.0,
            "official_metric_seconds": 0.2,
            "model_bytes": 1000,
        },
    ]

    summary = _rate_summaries(rows)[0]

    assert summary["mean_stream_bytes"] == 50.0
    assert summary["mean_actual_bpp"] == pytest.approx(0.1953125)
    assert summary["mean_actual_bpov"] == pytest.approx(0.25)
    assert summary["official_d1_rmse_grid_units"] == pytest.approx(26**0.5)
    assert summary["official_d2_rmse_grid_units"] == pytest.approx(17**0.5)


def test_retrained_model_size_excludes_training_events(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint").write_text('model_checkpoint_path: "model.ckpt-5000"\n')
    (checkpoint / "model.ckpt-5000.index").write_bytes(b"index")
    (checkpoint / "model.ckpt-5000.data-00000-of-00001").write_bytes(b"weights")
    (checkpoint / "train").mkdir()
    (checkpoint / "train" / "events").write_bytes(b"not deployment data")

    assert _trained_checkpoint_bytes(checkpoint) == len(
        (checkpoint / "checkpoint").read_bytes()
    ) + len(b"indexweights")
