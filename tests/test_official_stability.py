from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from pointconstellation.official_stability import (
    OfficialStabilityConfig,
    _bootstrap_comparison,
    _contains_forbidden_key,
    summarize_official_rows,
)


def test_official_config_rejects_partial_factorial() -> None:
    with pytest.raises(ValueError, match="at least two unique"):
        OfficialStabilityConfig(decoder_seeds=(7,))
    with pytest.raises(ValueError, match="validation and/or ood"):
        OfficialStabilityConfig(splits=("test",))


def test_selection_key_audit_rejects_test_fields() -> None:
    assert not _contains_forbidden_key({"selection": {"calibration_score": 1.0}})
    assert _contains_forbidden_key({"selection": {"validation": 1.0}})


def test_crossed_bootstrap_detects_consistent_official_gain() -> None:
    fps = np.asarray(
        [[4.0, 9.0, 16.0, 25.0], [5.0, 10.0, 17.0, 26.0]],
        dtype=np.float64,
    )
    refiner = np.stack((fps * 0.64, fps * 0.70), axis=1)
    result = _bootstrap_comparison(
        fps,
        refiner,
        np.asarray(["a", "a", "b", "b"]),
        samples=500,
        confidence_level=0.95,
        seed=17,
    )

    assert result["every_decoder_better_than_fps"]
    assert result["passes_positive_interval"]
    assert result["confidence_interval_lower_percent"] > 0.0


def test_official_row_summary_requires_d1_and_d2_positive() -> None:
    config = replace(
        OfficialStabilityConfig(),
        decoder_seeds=(7, 17),
        refiner_seeds=(101, 211),
        splits=("validation",),
        bootstrap_samples=200,
    )
    rows = []
    for decoder_seed in config.decoder_seeds:
        for sample_id, family in enumerate(("a", "a", "b", "b")):
            base = float(10 + sample_id + decoder_seed / 100)
            common = {
                "split": "validation",
                "decoder_seed": decoder_seed,
                "family": family,
                "model_id": f"{family}_{sample_id}",
                "sample_id": sample_id,
            }
            rows.append(
                {
                    **common,
                    "method": "fps",
                    "refiner_seed": None,
                    "d1_mse": base,
                    "d2_mse": 0.5 * base,
                }
            )
            for refiner_seed in config.refiner_seeds:
                rows.append(
                    {
                        **common,
                        "method": "refiner",
                        "refiner_seed": refiner_seed,
                        "d1_mse": 0.7 * base,
                        "d2_mse": 0.4 * base,
                    }
                )

    summary = summarize_official_rows(rows, config)

    assert summary["official_metric_gate_passes"]
    assert {row["metric"] for row in summary["comparisons"]} == {
        "d1_mse",
        "d2_mse",
    }
