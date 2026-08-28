from __future__ import annotations

from typing import Any

from pointconstellation.resummarize_geometry_metrics import (
    GeometryResummarizeConfig,
    summarize_geometry_rows,
)


def _row(
    *,
    method: str,
    decoder_seed: int,
    sample_id: int,
    refiner_seed: int | None,
    scale: float,
    budget: int | None = None,
) -> dict[str, Any]:
    return {
        "split": "validation",
        "method": method,
        "decoder_seed": decoder_seed,
        "refiner_seed": refiner_seed,
        "budget": budget,
        "family": "a" if sample_id < 2 else "b",
        "model_id": f"mesh_{sample_id}",
        "sample_id": sample_id,
        "surface_mse": 0.04 * scale,
        "normal_consistency": 1.0 - 0.2 * scale,
        "fresh_p90_euclidean": 0.2 * scale,
        "fresh_p99_euclidean": 0.3 * scale,
        "fresh_hausdorff": 0.4 * scale,
    }


def test_geometry_resummary_compares_complete_surface_cells() -> None:
    rows = []
    for decoder_seed in (7, 17):
        for sample_id in range(4):
            rows.append(
                _row(
                    method="fps",
                    decoder_seed=decoder_seed,
                    sample_id=sample_id,
                    refiner_seed=None,
                    scale=1.0,
                )
            )
            rows.append(
                _row(
                    method="random_best_of_16",
                    decoder_seed=decoder_seed,
                    sample_id=sample_id,
                    refiner_seed=None,
                    scale=0.9,
                )
            )
            for refiner_seed in (101, 211):
                rows.append(
                    _row(
                        method="refiner",
                        decoder_seed=decoder_seed,
                        sample_id=sample_id,
                        refiner_seed=refiner_seed,
                        scale=0.7,
                    )
                )
            rows.append(
                _row(
                    method="adam_multistart",
                    decoder_seed=decoder_seed,
                    sample_id=sample_id,
                    refiner_seed=None,
                    budget=16,
                    scale=0.6,
                )
            )
    config = GeometryResummarizeConfig(splits=("validation",), bootstrap_samples=100)

    result = summarize_geometry_rows(rows, config, decoder_seeds=(7, 17))

    complete = [
        row
        for row in result["comparisons"]
        if row["status"] == "complete" and row["metric"] == "surface_mse"
    ]
    assert {row["candidate_arm"] for row in complete} == {
        "random_best_of_16",
        "refiner",
        "adam_multistart:budget_16",
    }
    assert all(row["effect"] > 0.0 for row in complete)
