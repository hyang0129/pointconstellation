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


def test_published_codec_config_validates_diversity_fraction() -> None:
    config = PublishedCodecBenchmarkConfig(
        codec_manifest="codec.json",
        stability_config="stability.json",
        pc_error_executable="pc_error",
    )

    assert config.minimum_unique_reconstruction_fraction == 0.9
    with pytest.raises(ValueError, match="minimum_unique_reconstruction_fraction"):
        PublishedCodecBenchmarkConfig(
            codec_manifest="codec.json",
            stability_config="stability.json",
            pc_error_executable="pc_error",
            minimum_unique_reconstruction_fraction=0.0,
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
            "stream_sha256": "1" * 64,
            "reconstruction_sha256": "2" * 64,
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
            "stream_sha256": "3" * 64,
            "reconstruction_sha256": "4" * 64,
        },
    ]

    summary = _rate_summaries(rows)[0]

    assert summary["mean_stream_bytes"] == 50.0
    assert summary["mean_actual_bpp"] == pytest.approx(0.1953125)
    assert summary["mean_actual_bpov"] == pytest.approx(0.25)
    assert summary["official_d1_rmse_grid_units"] == pytest.approx(26**0.5)
    assert summary["official_d2_rmse_grid_units"] == pytest.approx(17**0.5)
    assert summary["unique_streams"] == 2
    assert summary["unique_reconstructions"] == 2
    assert summary["constant_output"] is False
    assert summary["rate_point_valid"] is True


def test_rate_summary_rejects_constant_stream_and_reconstruction_hashes() -> None:
    rows = [
        {
            "lambda": "1.00e-05",
            "split": "validation",
            "stream_bytes": 46,
            "actual_stream_bpp": 0.1796875,
            "actual_stream_bpov": 0.2,
            "chamfer_mse": 1.0,
            "d1_mse": 1.0,
            "d2_mse": 1.0,
            "d1_psnr_db": 1.0,
            "d2_psnr_db": 1.0,
            "encode_seconds": 1.0,
            "official_metric_seconds": 1.0,
            "model_bytes": 1000,
            "stream_sha256": "1" * 64,
            "reconstruction_sha256": "2" * 64,
        }
        for _ in range(10)
    ]

    summary = _rate_summaries(rows)[0]

    assert summary["unique_streams"] == 1
    assert summary["unique_reconstructions"] == 1
    assert summary["constant_output"] is True
    assert summary["rate_point_valid"] is False


def test_rate_summary_applies_configurable_reconstruction_diversity_threshold() -> None:
    rows = []
    for index in range(10):
        reconstruction_index = index if index < 8 else 0
        rows.append(
            {
                "lambda": "1.00e-05",
                "split": "ood",
                "stream_bytes": 46,
                "actual_stream_bpp": 0.1796875,
                "actual_stream_bpov": 0.2,
                "chamfer_mse": 1.0,
                "d1_mse": 1.0,
                "d2_mse": 1.0,
                "d1_psnr_db": 1.0,
                "d2_psnr_db": 1.0,
                "encode_seconds": 1.0,
                "official_metric_seconds": 1.0,
                "model_bytes": 1000,
                "stream_sha256": f"{index:064x}",
                "reconstruction_sha256": f"{reconstruction_index:064x}",
            }
        )

    default_summary = _rate_summaries(rows)[0]
    relaxed_summary = _rate_summaries(rows, minimum_unique_reconstruction_fraction=0.8)[
        0
    ]

    assert default_summary["unique_reconstructions"] == 8
    assert default_summary["constant_output"] is True
    assert default_summary["rate_point_valid"] is False
    assert relaxed_summary["constant_output"] is False
    assert relaxed_summary["rate_point_valid"] is True


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
