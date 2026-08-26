from __future__ import annotations

from typing import Any

import pytest

from pointconstellation.bd_rate import (
    aggregate_rd_curve,
    bd_psnr,
    bd_rate,
    bootstrap_bd_comparison,
)


def test_bjontegaard_matches_published_campfire_example() -> None:
    # Bull and Zhang, Intelligent Image and Video Compression, 2nd ed.,
    # Example 4.6. The published rounded results are 1.292 dB and -52.64%.
    anchor_rate = [5.947, 11.393, 23.844, 53.758]
    anchor_psnr = [33.720, 35.467, 36.789, 37.894]
    test_rate = [4.925, 8.965, 17.556, 59.923]
    test_psnr = [35.016, 36.420, 37.499, 39.272]

    rate = bd_rate(anchor_rate, anchor_psnr, test_rate, test_psnr)
    psnr = bd_psnr(anchor_rate, anchor_psnr, test_rate, test_psnr)

    assert rate.reason is None
    assert rate.value == pytest.approx(-52.6393, abs=1e-4)
    assert psnr.reason is None
    assert psnr.value == pytest.approx(1.29171, abs=1e-5)


def test_bjontegaard_gates_too_few_points_and_nonoverlap() -> None:
    too_short = bd_rate([1, 2, 3], [30, 31, 32], [1, 2, 3], [31, 32, 33])
    no_overlap = bd_rate(
        [1, 2, 3, 4],
        [20, 21, 22, 23],
        [1, 2, 3, 4],
        [30, 31, 32, 33],
    )

    assert too_short.value is None
    assert "at least 4" in str(too_short.reason)
    assert no_overlap.value is None
    assert "no positive quality overlap" in str(no_overlap.reason)


def _registry_rows(method: str, rate_scale: float) -> list[dict[str, Any]]:
    rows = []
    refiner_seeds = (101, 211) if method == "refiner" else (None,)
    for decoder_seed in (7, 17):
        for refiner_seed in refiner_seeds:
            for sample_id in range(4):
                for point, rate in enumerate((0.1, 0.2, 0.4, 0.8)):
                    rows.append(
                        {
                            "record_kind": "per_cloud",
                            "dataset": "FixtureNet",
                            "split": "validation",
                            "method": method,
                            "arm_label": "stabilized",
                            "experiment": "experiment_025_fixture",
                            "rate_point": f"r{point}",
                            "rate_bpp": rate * rate_scale,
                            "decoder_seed": decoder_seed,
                            "refiner_seed": refiner_seed,
                            "pair_id": f"cloud-{sample_id}",
                            "family": "fixture",
                            "model_id": f"model-{sample_id}",
                            "sample_id": sample_id,
                            "metric_name": "official_d1_psnr_db",
                            "value": 30.0 + 2.0 * point + 0.1 * sample_id,
                        }
                    )
    return rows


def test_seed_cloud_bootstrap_is_deterministic_and_marks_dominated_points() -> None:
    anchor = _registry_rows("feature_latent", 1.0)
    test = _registry_rows("refiner", 0.8)

    comparison = bootstrap_bd_comparison(
        anchor,
        test,
        rate_field="rate_bpp",
        bootstrap_samples=40,
        bootstrap_seed=19,
    )
    repeated = bootstrap_bd_comparison(
        anchor,
        test,
        rate_field="rate_bpp",
        bootstrap_samples=40,
        bootstrap_seed=19,
    )
    curve = aggregate_rd_curve(
        test,
        rate_field="rate_bpp",
        bootstrap_samples=20,
        bootstrap_seed=23,
    )
    dominated_rows = _registry_rows("refiner", 0.8)
    for row in dominated_rows:
        if row["rate_point"] == "r2":
            row["value"] = 31.0
    curve_with_dominated_point = aggregate_rd_curve(
        dominated_rows,
        rate_field="rate_bpp",
        bootstrap_samples=0,
    )

    assert comparison == repeated
    assert comparison.paired_clouds == 4
    assert comparison.bd_rate.value == pytest.approx(-20.0)
    assert comparison.bd_rate.ci_lower == pytest.approx(-20.0)
    assert comparison.bd_rate.ci_upper == pytest.approx(-20.0)
    assert len(curve.points) == 4
    assert not any(point.dominated for point in curve.points)
    dominated = {
        point.rate_point: point.dominated for point in curve_with_dominated_point.points
    }
    assert dominated["r2"]
