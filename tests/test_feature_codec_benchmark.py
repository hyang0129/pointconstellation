# ruff: noqa: E402

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from pointconstellation.feature_codec_benchmark import (
    FeatureCodecBenchmarkConfig,
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
        assert set(seed_result["model"]["decoder_state_dicts"]) == {"fp32", "fp16"}
        assert all(
            set(row["amortized_bpp"]) == {"fp32", "fp16"}
            for row in seed_result["summary"]
        )
        for row in seed_result["per_cloud"]:
            expected_bytes = {2: 28, 4: 34}[row["constellation_size"]]
            assert row["stream_bytes"] == expected_bytes
            assert row["actual_stream_bpp"] == expected_bytes * 8 / 32
            assert (
                row["header_bytes"] + row["payload_bytes"] + row["normalization_bytes"]
                == expected_bytes
            )
            assert row["payload_bpp"] == row["payload_bytes"] * 8 / 32
            assert row["fresh_chamfer_mse"] >= 0
