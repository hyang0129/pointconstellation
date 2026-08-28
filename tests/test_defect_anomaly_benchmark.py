"""Focused tests for the Experiment 041 anomaly benchmark."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from pointconstellation.defect_anomaly_benchmark import (
    DefectAnomalyBenchmarkConfig,
    KNNNormalManifoldScorer,
    KNNScorerConfig,
    PointCloudCodec,
    assert_matched_bytes,
    binary_auprc,
    binary_auroc,
    build_diagnostic_subset_codec_arms,
    load_external_anomaly_manifest,
)
from pointconstellation.defects import DEFECT_TYPES, inject_defect


def _sphere(count: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(71)
    z = rng.uniform(-1.0, 1.0, count)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = np.sqrt(1.0 - z * z)
    normals = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    return (0.7 * normals).astype(np.float32), normals.astype(np.float32)


def test_nonlearned_scorer_exceeds_raw_smoke_auroc() -> None:
    normal, normals = _sphere()
    scorer = KNNNormalManifoldScorer(
        KNNScorerConfig(
            points_per_reference=len(normal),
            maximum_reference_clouds=1,
            cloud_candidates=1,
        ),
        seed=73,
    ).fit([normal])
    labels = []
    scores = []
    for index, defect_type in enumerate(DEFECT_TYPES):
        defect = inject_defect(
            normal,
            defect_type,
            seed=80 + index,
            fraction=0.03,
            normals=normals,
        )
        labels.extend(defect.point_labels.tolist())
        scores.extend(scorer.score_points(defect.points).tolist())

    assert binary_auroc(labels, scores) > 0.9


def test_binary_metrics_handle_ties_deterministically() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.0, 1.0, 0.0, 1.0]

    assert binary_auroc(labels, scores) == 1.0
    assert binary_auprc(labels, scores) == 1.0
    assert np.isnan(binary_auroc([0, 0], [0.0, 1.0]))


def test_matched_bytes_assertion_uses_actual_serialized_lengths() -> None:
    assert (
        assert_matched_bytes(
            {"constellation_only": bytes(50), "selective_pass_through": bytes(50)},
            target_bytes=50,
        )
        == 50
    )
    with pytest.raises(AssertionError, match="not byte matched"):
        assert_matched_bytes(
            {"constellation_only": bytes(49), "selective_pass_through": bytes(50)}
        )


def test_codec_protocol_excludes_labels_and_diagnostic_streams_match() -> None:
    config = DefectAnomalyBenchmarkConfig.from_json(
        Path("configs/experiment_041_defect_anomaly_smoke.json")
    )
    arms = build_diagnostic_subset_codec_arms(config)
    points, _ = _sphere(256)

    assert set(inspect.signature(PointCloudCodec.encode).parameters) == {
        "self",
        "source",
    }
    assert set(inspect.signature(PointCloudCodec.decode).parameters) == {
        "self",
        "stream",
    }
    for payload_budget in config.payload_budgets:
        expected_bytes = 16 + payload_budget
        streams = {
            arm.name: arm.codec.encode(points)
            for arm in arms
            if arm.payload_budget_bytes == payload_budget
        }
        assert_matched_bytes(streams, target_bytes=expected_bytes)
        for arm in (row for row in arms if row.payload_budget_bytes == payload_budget):
            assert isinstance(arm.codec, PointCloudCodec)
            assert arm.codec.decode(streams[arm.name]).shape[1] == 3


def test_smoke_and_full_configs_are_valid_and_use_three_scorer_seeds() -> None:
    smoke = DefectAnomalyBenchmarkConfig.from_json(
        Path("configs/experiment_041_defect_anomaly_smoke.json")
    )
    full = DefectAnomalyBenchmarkConfig.from_json(
        Path("configs/experiment_041_defect_anomaly.json")
    )

    assert smoke.diagnostic_subset_codecs
    assert not full.diagnostic_subset_codecs
    assert len(smoke.scorer_seeds) == len(full.scorer_seeds) == 3
    assert smoke.payload_budgets == full.payload_budgets == (40, 52, 64, 78, 96, 110)


def test_external_dataset_manifest_hook_validates_without_downloading(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mvtec_manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": "MVTec 3D-AD",
                "splits": {
                    "test": [
                        {
                            "model_id": "bagel_001",
                            "category": "bagel",
                            "pointcloud": "bagel/test/001.npz",
                            "pointcloud_sha256": "0" * 64,
                            "cloud_label": 1,
                        }
                    ]
                },
            }
        )
    )

    manifest = load_external_anomaly_manifest(path)

    assert manifest["dataset"] == "MVTec 3D-AD"
    assert not (tmp_path / "bagel").exists()
