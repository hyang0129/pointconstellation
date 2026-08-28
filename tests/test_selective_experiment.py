"""Focused no-training runner tests for Experiment 040."""

# ruff: noqa: E402

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.codecs import OfficialMetricResult
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.selective_experiment import (
    RatePoint,
    SelectiveExperimentConfig,
    _rate_points,
    run_selective_experiment,
)
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _decoder,
)


def _write_tiny_artifact(
    root: Path,
    stability_path: Path,
    stability: StabilityExperimentConfig,
) -> None:
    artifact = root / "stability"
    artifact.mkdir()
    datasets = _datasets(stability)
    stability_path.write_text(json.dumps(asdict(stability), indent=2) + "\n")
    (artifact / "stability_metrics.json").write_text(
        json.dumps(
            {
                "config": asdict(stability),
                "contract_checks": {"tiny_fixture": True},
                "data_protocol": _data_protocol(stability, datasets),
            },
            indent=2,
        )
        + "\n"
    )
    device = torch.device("cpu")
    for seed in stability.decoder_seeds:
        torch.manual_seed(seed)
        decoder = _decoder(stability, device)
        state_hash = _state_hash(decoder)
        decoder_dir = artifact / "decoders" / f"seed_{seed}"
        decoder_dir.mkdir(parents=True)
        (decoder_dir / "selection.json").write_text(
            json.dumps({"arms": {"stabilized": {"state_hash": state_hash}}}) + "\n"
        )
        torch.save(
            {
                "model": decoder.state_dict(),
                "state_hash": state_hash,
                "selection": {"selected_state_hash": state_hash},
            },
            decoder_dir / "stabilized.pt",
        )


def test_rate_grid_has_six_payload_points_and_26_byte_q8_diagnostic() -> None:
    rates = _rate_points(SelectiveExperimentConfig(decoder_seeds=(7,)))

    assert [rate.payload_budget_bytes for rate in rates[:-1]] == [
        40,
        52,
        64,
        78,
        96,
        110,
    ]
    assert [rate.constellation_size for rate in rates[:-1]] == [8, 11, 14, 17, 21, 24]
    assert rates[-1] == RatePoint(
        label="stream_26_q_8",
        coordinate_bits=8,
        constellation_size=4,
        payload_budget_bytes=12,
        fixed_payload_bytes=12,
        diagnostic_uniform_only=True,
    )


def test_tiny_selective_runner_writes_metrics_and_pending_gate(
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
    _write_tiny_artifact(tmp_path, stability_path, stability)
    executable = tmp_path / "pc_error"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)

    def fake_pc_error(*args, **kwargs) -> OfficialMetricResult:
        reconstruction = np.asarray(args[2])
        mse = 1.0 + float(np.mean(reconstruction**2))
        return OfficialMetricResult(
            metrics={"d1_mse": mse, "d2_mse": 0.5 * mse},
            elapsed_seconds=0.001,
            command=("fixture",),
            stdout="",
        )

    monkeypatch.setattr(
        "pointconstellation.selective_experiment.run_pc_error", fake_pc_error
    )
    config = SelectiveExperimentConfig(
        stability_config=str(stability_path),
        stability_artifact_dir=str(tmp_path / "stability"),
        pc_error_executable=str(executable),
        gpcc_reference_path=None,
        position_bits=8,
        payload_budgets=(12,),
        coordinate_bits=8,
        diagnostic_q8_stream_bytes=None,
        decoder_seeds=(1,),
        adam_evaluations=2,
        irregularity_neighbors=3,
        irregularity_chunk_size=8,
        minimum_spacing=0.0,
        normal_neighbors=3,
        compute_normal_consistency=False,
        splits=("validation",),
        max_clouds_per_split=2,
        batch_size=2,
        bootstrap_samples=100,
        output_dir=str(tmp_path / "selective"),
    )

    result = run_selective_experiment(config, device_name="cpu")
    rows = [
        json.loads(line)
        for line in (tmp_path / "selective" / "selective_per_cloud.jsonl")
        .read_text()
        .splitlines()
    ]

    assert result["rows"] == len(rows) == 32
    assert result["gate_g_c1"]["evaluable"] is False
    assert result["gate_g_c1"]["passes"] is None
    assert all(result["contract_checks"].values())
    assert all(row["source_only_selection"] for row in rows)
    assert all(row["preservation_error"] in {None, 0.0} for row in rows)
    assert all(sum(cell["count"] for cell in row["stratified_d1"]) == 8 for row in rows)
    assert (tmp_path / "selective" / "run_manifest.json").is_file()
    assert (tmp_path / "selective" / "selective_metrics.json").is_file()
