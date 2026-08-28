"""Focused tests for Experiment 033 encoder-objective sweep."""

# ruff: noqa: E402

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.codecs import OfficialMetricResult
from pointconstellation.objective_sweep_experiment import (
    OBJECTIVE_NAMES,
    OBJECTIVE_REGISTRY,
    ObjectiveContext,
    ObjectiveSweepExperimentConfig,
    ObjectiveSweepRegime,
    PointNetClassifier,
    _encoding_identity,
    _selection_seed,
    _source_tensor,
    build_objective_scorer,
    estimate_source_normals_pca,
    run_objective_sweep_experiment,
)
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _decoder,
)


class _MeanDecoder(torch.nn.Module):
    def forward(
        self, coordinates: torch.Tensor, *, num_output_points: int
    ) -> torch.Tensor:
        repeats = (num_output_points + coordinates.shape[1] - 1) // coordinates.shape[1]
        return coordinates.repeat(1, repeats, 1)[:, :num_output_points]


def _objective_context(source: torch.Tensor) -> ObjectiveContext:
    decoder = _MeanDecoder().eval().requires_grad_(False)
    torch.manual_seed(13)
    classifier = (
        PointNetClassifier(3, feature_width=8, feature_dim=8)
        .eval()
        .requires_grad_(False)
    )
    return ObjectiveContext(
        decoder=decoder,
        source_points=source,
        num_output_points=source.shape[1],
        distance_chunk_size=4,
        normal_neighbors=3,
        normal_chunk_size=4,
        mixed_chamfer_weight=0.5,
        feature_extractor=classifier,
    )


@pytest.mark.parametrize("objective", OBJECTIVE_NAMES)
def test_each_registered_objective_is_differentiable(objective: str) -> None:
    torch.manual_seed(7)
    source = torch.rand(2, 8, 3) * 1.5 - 0.75
    coordinates = (torch.rand(2, 3, 3) - 0.5).requires_grad_(True)

    losses = build_objective_scorer(objective, _objective_context(source))(coordinates)
    losses.mean().backward()

    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
    assert float(coordinates.grad.abs().sum()) > 0.0


def test_registry_is_complete_and_rejects_unknown_objectives() -> None:
    assert tuple(OBJECTIVE_REGISTRY) == OBJECTIVE_NAMES
    source = torch.rand(1, 8, 3)

    with pytest.raises(ValueError, match="unknown objective"):
        build_objective_scorer("target_assisted", _objective_context(source))


class _AccessAudit(Mapping[str, Any]):
    def __init__(self) -> None:
        self.values = {
            "source_points": torch.rand(8, 3),
            "fresh_points": torch.full((8, 3), 99.0),
            "source_normals": torch.full((8, 3), 88.0),
            "label": "forbidden",
            "category": "forbidden",
        }
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> Any:
        self.accessed.append(key)
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def get(self, key: str, default: Any = None) -> Any:
        self.accessed.append(key)
        return self.values.get(key, default)


def test_encode_input_and_objective_context_structurally_exclude_targets() -> None:
    sample = _AccessAudit()

    assert _source_tensor(sample) is sample.values["source_points"]
    assert set(sample.accessed) == {"source_points"}
    sample.accessed.clear()
    assert _encoding_identity(sample, 17) == {"model_id": "17", "sample_id": 17}
    assert set(sample.accessed) == {"model_id", "sample_id"}
    assert {field.name for field in fields(ObjectiveContext)} == {
        "decoder",
        "source_points",
        "num_output_points",
        "distance_chunk_size",
        "normal_neighbors",
        "normal_chunk_size",
        "mixed_chamfer_weight",
        "feature_extractor",
    }


def test_pca_normals_are_source_derived_detached_and_unit_length() -> None:
    torch.manual_seed(19)
    source = torch.rand(2, 9, 3, requires_grad=True)

    normals = estimate_source_normals_pca(source, neighbors=4, chunk_size=3)

    assert normals.shape == source.shape
    assert not normals.requires_grad
    assert torch.allclose(torch.linalg.vector_norm(normals, dim=2), torch.ones(2, 9))


def test_pointnet_features_are_permutation_invariant() -> None:
    torch.manual_seed(23)
    classifier = PointNetClassifier(4, feature_width=8, feature_dim=12).eval()
    points = torch.rand(2, 11, 3)

    first = classifier.extract_features(points)
    second = classifier.extract_features(points[:, torch.randperm(11)])

    assert torch.allclose(first, second, atol=1e-6)


def test_selection_seed_is_source_only_and_permutation_invariant() -> None:
    config = ObjectiveSweepExperimentConfig(
        regimes=(
            ObjectiveSweepRegime(
                "tiny",
                8,
                2,
                "unused.json",
                "unused-artifact",
            ),
        ),
        pc_error_executable="unused-pc-error",
        decoder_seeds=(1, 2),
        start_methods=("fps",),
        normal_neighbors=3,
        classifier_regime="tiny",
        bootstrap_samples=100,
    )
    source = torch.rand(8, 3)

    first = _selection_seed(config, regime="tiny", method="fps", source_points=source)
    second = _selection_seed(
        config,
        regime="tiny",
        method="fps",
        source_points=source[torch.randperm(8)],
    )

    assert first == second


def _write_tiny_artifact(
    root: Path,
    stability_path: Path,
    stability: StabilityExperimentConfig,
) -> Path:
    artifact = root / "stability"
    artifact.mkdir()
    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    stability_path.write_text(json.dumps(asdict(stability), indent=2) + "\n")
    (artifact / "stability_metrics.json").write_text(
        json.dumps(
            {
                "config": asdict(stability),
                "contract_checks": {"tiny_fixture": True},
                "data_protocol": data_protocol,
            },
            indent=2,
        )
        + "\n"
    )
    device = torch.device("cpu")
    for decoder_seed in stability.decoder_seeds:
        torch.manual_seed(decoder_seed)
        decoder = _decoder(stability, device)
        decoder_hash = _state_hash(decoder)
        decoder_dir = artifact / "decoders" / f"seed_{decoder_seed}"
        decoder_dir.mkdir(parents=True)
        selection = {"arms": {"stabilized": {"state_hash": decoder_hash}}}
        (decoder_dir / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
        torch.save(
            {
                "model": decoder.state_dict(),
                "state_hash": decoder_hash,
                "selection": {"selected_state_hash": decoder_hash},
            },
            decoder_dir / "stabilized.pt",
        )
    return artifact


def test_tiny_objective_sweep_runs_end_to_end_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stability_path = tmp_path / "stability.json"
    stability = StabilityExperimentConfig(
        num_points=8,
        constellation_size=2,
        training_constellation_sizes=(2, 4),
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
        feature_width=8,
        num_heads=2,
        distance_chunk_size=8,
        bootstrap_samples=100,
        output_dir=str(tmp_path / "unused"),
    )
    artifact = _write_tiny_artifact(tmp_path, stability_path, stability)
    executable = tmp_path / "pc_error"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    calls = 0

    def fake_pc_error(*args, **kwargs) -> OfficialMetricResult:
        nonlocal calls
        calls += 1
        reconstruction = np.asarray(args[2])
        mse = 1.0 + float(np.mean(reconstruction**2))
        return OfficialMetricResult(
            metrics={"d1_mse": mse, "d2_mse": 0.5 * mse},
            elapsed_seconds=0.001,
            command=("fixture",),
            stdout="",
        )

    monkeypatch.setattr(
        "pointconstellation.objective_sweep_experiment.run_pc_error", fake_pc_error
    )
    config = ObjectiveSweepExperimentConfig(
        regimes=(
            ObjectiveSweepRegime(
                name="tiny",
                num_points=8,
                constellation_size=2,
                stability_config=str(stability_path),
                stability_artifact_dir=str(artifact),
            ),
        ),
        pc_error_executable=str(executable),
        required_coordinate_bits=8,
        position_bits=8,
        decoder_seeds=(1, 2),
        start_methods=("fps",),
        decoder_evaluation_budget=2,
        normal_neighbors=3,
        normal_chunk_size=4,
        splits=("validation",),
        max_clouds_per_split=2,
        batch_size=2,
        classifier_regime="tiny",
        classifier_epochs=1,
        classifier_feature_width=8,
        classifier_feature_dim=8,
        bootstrap_samples=100,
        gate_min_source_accuracy=0.0,
        gate_min_regimes=1,
        output_dir=str(tmp_path / "objective_sweep"),
    )

    result = run_objective_sweep_experiment(config, device_name="cpu")
    first_call_count = calls
    resumed = run_objective_sweep_experiment(config, device_name="cpu")
    rows = [
        json.loads(line)
        for line in (tmp_path / "objective_sweep" / "objective_sweep_per_cloud.jsonl")
        .read_text()
        .splitlines()
    ]

    expected_rows = 2 * 2 * 4
    assert result["per_cloud_rows"] == expected_rows
    assert resumed["resumed_rows"] == expected_rows
    assert resumed["per_cloud_rows"] == expected_rows
    assert calls == first_call_count
    assert set(result["contract_checks"].values()) == {True}
    assert {row["objective"] for row in rows} == set(OBJECTIVE_NAMES)
    assert all(row["source_only_optimization"] for row in rows)
    assert all(row["labels_used_after_encoding_only"] for row in rows)
    assert all(row["provided_normals_used_after_encoding_only"] for row in rows)
    assert all(row["actual_stream_bits"] == 160 for row in rows)
    assert len(result["statistics"]["pareto_tables"]) == 4
    assert len(result["statistics"]["g_b2"]["comparisons"]) == 3


def test_checked_in_configs_cover_smoke_and_full_factorial() -> None:
    smoke = ObjectiveSweepExperimentConfig.from_json(
        Path("configs/experiment_033_objective_sweep_smoke.json")
    )
    full = ObjectiveSweepExperimentConfig.from_json(
        Path("configs/experiment_033_objective_sweep.json")
    )

    assert smoke.objectives == OBJECTIVE_NAMES
    assert {
        (regime.constellation_size, regime.num_points) for regime in full.regimes
    } == {(k, n) for k in (4, 8, 16, 32) for n in (1024, 2048, 4096)}
    assert full.required_coordinate_bits == 12
