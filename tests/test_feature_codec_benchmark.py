# ruff: noqa: E402

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from pointconstellation.bitstream import expected_stream_bytes
from pointconstellation.feature_bitstream import expected_feature_stream_bytes
from pointconstellation.feature_codec_benchmark import (
    FeatureCodecBenchmarkConfig,
    _data_protocol,
    _datasets,
    _stability_reference_comparison,
    run_feature_codec_benchmark,
)


def test_feature_codec_smoke_has_exact_rates_and_independent_models(
    tmp_path: Path,
) -> None:
    config = replace(
        FeatureCodecBenchmarkConfig.from_json(
            Path("configs/experiment_018_feature_codec_smoke.json")
        ),
        output_dir=str(tmp_path / "feature"),
    )

    result = run_feature_codec_benchmark(config, device_name="cpu")

    assert result["model_independence"]["all_unique"]
    assert len(result["per_seed"]) == 2
    assert result["matched_rate_comparisons"] == []
    assert not result["primary_gate"]["passes"]
    for seed_result in result["per_seed"]:
        assert seed_result["model"]["optimizer_updates"] == 2
        assert len(seed_result["summary"]) == 4
        for row in seed_result["per_cloud"]:
            expected_bytes = {2: 20, 4: 26}[row["constellation_size"]]
            assert row["stream_bytes"] == expected_bytes
            assert row["actual_stream_bpp"] == expected_bytes * 8 / 32
            assert row["fresh_chamfer_mse"] >= 0


def test_experiment_023_declares_all_four_exact_byte_matches() -> None:
    config = FeatureCodecBenchmarkConfig.from_json(
        Path("configs/experiment_023_feature_codec_equal_protocol.json")
    )

    assert config.matched_constellation_sizes == (4, 8, 16, 32)
    assert len(config.model_seeds) == 6
    matched_bytes = []
    for latent_dim, constellation_size in zip(
        config.latent_dims, config.matched_constellation_sizes, strict=True
    ):
        feature_bytes = expected_feature_stream_bytes(latent_dim, config.feature_bits)
        coordinate_bytes = expected_stream_bytes(
            constellation_size, config.coordinate_bits
        )
        assert feature_bytes == coordinate_bytes
        matched_bytes.append(feature_bytes)
    assert matched_bytes == [32, 50, 86, 158]

    with pytest.raises(ValueError, match="equal stream bytes"):
        replace(config, latent_dims=(19, 38, 74, 146))


def test_experiment_023_smoke_exercises_ema_calibration_selection(
    tmp_path: Path,
) -> None:
    config = replace(
        FeatureCodecBenchmarkConfig.from_json(
            Path("configs/experiment_023_feature_codec_smoke.json")
        ),
        output_dir=str(tmp_path / "feature"),
    )

    result = run_feature_codec_benchmark(config, device_name="cpu")

    assert result["experiment"] == "023_feature_codec_equal_protocol"
    assert result["data_protocol"]["all_partitions_pairwise_disjoint"]
    assert result["contract_checks"]["all_ema_paths_selected_by_calibration"]
    assert result["contract_checks"]["all_stream_sizes_match_declared_rates"]
    assert [row["stream_bytes"] for row in result["protocol"]["rate_points"]] == [
        32,
        50,
        86,
        158,
    ]
    for seed_result in result["per_seed"]:
        model = seed_result["model"]
        assert model["optimizer_updates"] == 2
        assert len(model["calibration_candidates"]) == 1
        assert model["selection"]["kind"] == "ema"
        assert model["selection"]["epoch"] == 2
        assert model["selection"]["selected_state_hash"] == model["state_hash"]
        assert Path(model["selection_record"]).is_file()


def test_equal_protocol_bootstrap_uses_all_independent_stability_cells(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "stability"
    reference_dir.mkdir()
    config = replace(
        FeatureCodecBenchmarkConfig.from_json(
            Path("configs/experiment_023_feature_codec_smoke.json")
        ),
        reference_stability_dir=str(reference_dir),
        bootstrap_samples=100,
    )
    data_protocol = _data_protocol(config, _datasets(config))
    reference_partitions = {
        "train": data_protocol["partitions"]["train"],
        "calibration": data_protocol["partitions"]["calibration"],
        "validation": data_protocol["partitions"]["validation"],
        "ood": data_protocol["partitions"]["category_ood"],
    }
    reference_config = {
        "data_seed": config.data_seed,
        "num_points": config.num_points,
        "train_samples": config.train_samples,
        "calibration_samples": config.calibration_samples,
        "validation_samples": config.validation_samples,
        "ood_samples": config.category_ood_samples,
        "batch_size": config.batch_size,
        "training_constellation_sizes": list(config.matched_constellation_sizes),
        "constellation_size": config.primary_constellation_size,
        "coordinate_bits": config.coordinate_bits,
        "baseline_decoder_epochs": 1,
        "stabilized_decoder_epochs": config.epochs,
        "ema_decay": config.ema_decay,
        "decoder_seeds": [11, 13],
        "refiner_seeds": [17, 19],
    }
    (reference_dir / "stability_metrics.json").write_text(
        json.dumps(
            {
                "config": reference_config,
                "data_protocol": {
                    "manifest_sha256": data_protocol["manifest_sha256"],
                    "partitions": reference_partitions,
                },
                "factorial": {"complete": True},
            }
        )
    )
    cloud_ids = {
        "validation": (("000001", "validation_a"), ("000001", "validation_b")),
        "ood": (("000002", "ood_a"), ("000002", "ood_b")),
    }
    coordinate_rows = []
    for decoder_seed in reference_config["decoder_seeds"]:
        for refiner_seed in reference_config["refiner_seeds"]:
            for split, keys in cloud_ids.items():
                for family, model_id in keys:
                    coordinate_rows.append(
                        {
                            "arm": "stabilized",
                            "split": split,
                            "method": "refiner",
                            "decoder_seed": decoder_seed,
                            "refiner_seed": refiner_seed,
                            "family": family,
                            "model_id": model_id,
                            "stream_bytes": 50,
                            "chamfer_mse": 0.01,
                            "fresh_chamfer_mse": 0.011,
                        }
                    )
    (reference_dir / "per_cloud.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in coordinate_rows)
    )
    seed_results = []
    for model_seed in config.model_seeds:
        feature_rows = []
        for split, coordinate_split in (
            ("validation", "validation"),
            ("category_ood", "ood"),
        ):
            for family, model_id in cloud_ids[coordinate_split]:
                feature_rows.append(
                    {
                        "split": split,
                        "method": "feature_latent",
                        "constellation_size": 8,
                        "family": family,
                        "model_id": model_id,
                        "stream_bytes": 50,
                        "chamfer_mse": 0.04,
                        "fresh_chamfer_mse": 0.044,
                    }
                )
        seed_results.append({"model_seed": model_seed, "per_cloud": feature_rows})

    comparison = _stability_reference_comparison(config, seed_results, data_protocol)

    assert comparison["status"] == "complete"
    assert comparison["primary"]["coordinate_decoder_seeds"] == 2
    assert comparison["primary"]["coordinate_refiner_seeds"] == 2
    assert comparison["primary"]["feature_codec_seeds"] == 2
    assert comparison["primary"]["confidence_interval_lower_percent"] > 0
    assert comparison["representation_gate_passes"]
