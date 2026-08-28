# ruff: noqa: E402

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.bitstream import expected_stream_bytes
from pointconstellation.codecs import OfficialMetricResult
from pointconstellation.data import TriangleMesh
from pointconstellation.official_stability import (
    OfficialStabilityConfig,
    _bootstrap_comparison,
    _contains_forbidden_key,
    _evaluate_method,
    summarize_official_rows,
)
from pointconstellation.stability_experiment import StabilityExperimentConfig


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
    original_rows = [
        {
            **row,
            "original_frame_d1_mse": row["d1_mse"] * 1.001,
            "original_frame_d2_mse": row["d2_mse"] * 1.001,
        }
        for row in rows
    ]
    original_summary = summarize_official_rows(original_rows, config)
    assert {row["coordinate_frame"] for row in original_summary["comparisons"]} == {
        "shared_normalized",
        "original_mesh",
    }
    assert original_summary["official_metric_gate_passes"]

def test_selection_summary_compares_arm_with_fps_and_refiner() -> None:
    config = replace(
        OfficialStabilityConfig(),
        decoder_seeds=(7, 17),
        refiner_seeds=(101, 211),
        methods=("fps", "kmeans", "refiner"),
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
            for method, scale in (("fps", 1.0), ("kmeans", 0.9)):
                rows.append(
                    {
                        **common,
                        "method": method,
                        "refiner_seed": None,
                        "d1_mse": scale * base,
                        "d2_mse": 0.5 * scale * base,
                    }
                )
            for refiner_seed in config.refiner_seeds:
                rows.append(
                    {
                        **common,
                        "method": "refiner",
                        "refiner_seed": refiner_seed,
                        "d1_mse": 0.6 * base,
                        "d2_mse": 0.3 * base,
                    }
                )

    summary = summarize_official_rows(rows, config)

    assert summary["selection_baseline_gate_passes"]
    assert len(summary["comparisons"]) == 6
    assert {
        (row["baseline_method"], row["candidate_method"])
        for row in summary["comparisons"]
    } == {("fps", "refiner"), ("kmeans", "refiner"), ("fps", "kmeans")}


class _TinyDecoder(torch.nn.Module):
    def forward(
        self, constellation: torch.Tensor, *, num_output_points: int
    ) -> torch.Tensor:
        repeats = (
            num_output_points + constellation.shape[1] - 1
        ) // constellation.shape[1]
        return constellation.repeat(1, repeats, 1)[:, :num_output_points]


def test_two_selection_methods_run_through_official_evaluator(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stability = StabilityExperimentConfig(
        num_points=16,
        constellation_size=4,
        training_constellation_sizes=(4, 8),
        coordinate_bits=8,
        train_samples=4,
        calibration_samples=2,
        validation_samples=2,
        ood_samples=2,
        batch_size=2,
        decoder_seeds=(1, 2),
        refiner_seeds=(3, 4),
        baseline_decoder_epochs=1,
        stabilized_decoder_epochs=2,
        distance_chunk_size=8,
    )
    official = OfficialStabilityConfig(
        pc_error_executable="unused",
        position_bits=8,
        decoder_seeds=(1, 2),
        refiner_seeds=(3, 4),
        splits=("validation",),
        bootstrap_samples=100,
    )
    generator = torch.Generator().manual_seed(73)
    sample = {
        "source_points": 1.8 * torch.rand(16, 3, generator=generator) - 0.9,
        "source_normals": torch.zeros(16, 3),
        "family": "fixture",
        "model_id": "tiny",
        "sample_id": 0,
    }

    def fake_pc_error(*args, **kwargs) -> OfficialMetricResult:
        return OfficialMetricResult(
            metrics={"d1_mse": 1.0, "d2_mse": 0.5},
            elapsed_seconds=0.01,
            command=("fixture",),
            stdout="",
        )

    monkeypatch.setattr(
        "pointconstellation.official_stability.run_pc_error", fake_pc_error
    )
    rows = [
        _evaluate_method(
            stability,
            official,
            decoder=_TinyDecoder().eval(),
            refiner=None,
            decoder_seed=1,
            refiner_seed=None,
            split="validation",
            sample=sample,
            device=torch.device("cpu"),
            scratch_root=tmp_path,
            method=method,
        )
        for method in ("kmeans", "random_best_of_16")
    ]

    assert [row["method"] for row in rows] == ["kmeans", "random_best_of_16"]
    assert {row["representation_class"] for row in rows} == {
        "free-coordinate",
        "strict-subset",
    }
    assert all(row["stream_bytes"] == expected_stream_bytes(4, 8) for row in rows)
    assert all(row["encode_seconds"] >= 0.0 for row in rows)


def test_official_evaluator_exposes_optional_mesh_metrics(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stability = StabilityExperimentConfig(
        num_points=16,
        constellation_size=4,
        training_constellation_sizes=(4, 8),
        coordinate_bits=8,
        train_samples=4,
        calibration_samples=2,
        validation_samples=2,
        ood_samples=2,
        batch_size=2,
        decoder_seeds=(1, 2),
        refiner_seeds=(3, 4),
        baseline_decoder_epochs=1,
        stabilized_decoder_epochs=2,
        distance_chunk_size=8,
    )
    official = OfficialStabilityConfig(
        pc_error_executable="unused",
        position_bits=8,
        decoder_seeds=(1, 2),
        refiner_seeds=(3, 4),
        splits=("validation",),
        compute_mesh_metrics=True,
        mesh_metric_point_chunk_size=8,
        mesh_metric_triangle_chunk_size=1,
        mesh_normal_neighbors=6,
        bootstrap_samples=100,
    )
    axis = torch.linspace(-0.8, 0.8, 4)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    source = torch.stack((x.ravel(), y.ravel(), torch.zeros(16)), dim=1)
    sample = {
        "source_points": source,
        "source_normals": torch.tensor([[0.0, 0.0, 1.0]]).repeat(16, 1),
        "family": "plane",
        "model_id": "plane",
        "sample_id": 0,
    }
    mesh = TriangleMesh(
        vertices=np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )

    monkeypatch.setattr(
        "pointconstellation.official_stability.run_pc_error",
        lambda *args, **kwargs: OfficialMetricResult(
            metrics={"d1_mse": 1.0, "d2_mse": 0.5},
            elapsed_seconds=0.01,
            command=("fixture",),
            stdout="",
        ),
    )
    row = _evaluate_method(
        stability,
        official,
        decoder=_TinyDecoder().eval(),
        refiner=None,
        decoder_seed=1,
        refiner_seed=None,
        split="validation",
        sample=sample,
        device=torch.device("cpu"),
        scratch_root=tmp_path,
        method="fps",
        mesh=mesh,
    )

    # An even-sized signed fixed-width lattice has no exact zero level.
    assert row["surface_rmse"] == pytest.approx(1.0 / 255.0, abs=1e-9)
    assert 0.0 <= row["normal_consistency"] <= 1.0
    assert row["p90_euclidean"] >= 0.0
    assert len(bytes.fromhex(row["stream_hex"])) == row["stream_bytes"]
