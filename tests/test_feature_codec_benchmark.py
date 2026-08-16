from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
        for row in seed_result["per_cloud"]:
            expected_bytes = {2: 20, 4: 26}[row["constellation_size"]]
            assert row["stream_bytes"] == expected_bytes
            assert row["actual_stream_bpp"] == expected_bytes * 8 / 32
            assert row["fresh_chamfer_mse"] >= 0
