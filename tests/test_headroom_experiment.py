"""Focused tests for Experiment 022 inference headroom."""

# ruff: noqa: E402

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.bitstream import decode_constellation, expected_stream_bytes
from pointconstellation.codecs import OfficialMetricResult
from pointconstellation.headroom_experiment import (
    HeadroomExperimentConfig,
    _search_adam_start,
    _select_best_by_source_score,
    run_headroom_experiment,
)
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _decoder,
    _refiner,
)


def test_each_adam_start_respects_decoder_evaluation_budget() -> None:
    budget = 5
    for start_index in range(4):
        calls = 0
        initial = torch.full((2, 3, 3), 0.1 * start_index)

        def score(coordinates: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return coordinates.square().mean(dim=(1, 2))

        result = _search_adam_start(
            score,
            initial,
            bits=8,
            budget=budget,
            learning_rate=0.03,
        )

        assert calls <= budget
        assert result.decoder_evaluations_per_cloud == budget


class _AccessAudit(Mapping[str, Any]):
    def __init__(self, source: float, label: str) -> None:
        self.values = {
            "source_chamfer_mse": source,
            "fresh_chamfer_mse": -1000.0 * source,
            "d1_mse": -2000.0 * source,
            "d2_mse": -3000.0 * source,
            "label": label,
        }
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> Any:
        self.accessed.append(key)
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def test_multistart_selection_never_reads_fresh_or_official_metrics() -> None:
    candidates = [_AccessAudit(0.4, "first"), _AccessAudit(0.2, "second")]

    selected = _select_best_by_source_score(candidates)

    assert selected is candidates[1]
    assert all(
        set(candidate.accessed) == {"source_chamfer_mse"} for candidate in candidates
    )


def _write_tiny_artifact(
    root: Path,
    stability_path: Path,
    stability: StabilityExperimentConfig,
) -> None:
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
        for refiner_seed in stability.refiner_seeds:
            torch.manual_seed(refiner_seed)
            refiner = _refiner(stability, device)
            pair_dir = (
                artifact
                / "pairs"
                / "stabilized"
                / f"decoder_{decoder_seed}_refiner_{refiner_seed}"
            )
            pair_dir.mkdir(parents=True)
            torch.save(
                {
                    "model": refiner.state_dict(),
                    "decoder_seed": decoder_seed,
                    "refiner_seed": refiner_seed,
                    "decoder_state_hash": decoder_hash,
                },
                pair_dir / "refiner.pt",
            )


def test_tiny_fixture_runs_end_to_end_and_resumes(
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
        "pointconstellation.headroom_experiment.run_pc_error", fake_pc_error
    )
    config = HeadroomExperimentConfig(
        stability_config=str(stability_path),
        stability_artifact_dir=str(tmp_path / "stability"),
        pc_error_executable=str(executable),
        position_bits=8,
        decoder_seeds=(1, 2),
        refiner_seeds=(3, 4),
        random_start_seeds=(5, 7),
        budgets=(2,),
        splits=("validation",),
        max_clouds_per_split=2,
        batch_size=2,
        timing_devices=("cpu",),
        bootstrap_samples=100,
        output_dir=str(tmp_path / "headroom"),
    )

    result = run_headroom_experiment(config, device_name="cpu")
    first_call_count = calls
    resumed = run_headroom_experiment(config, device_name="cpu")
    rows = [
        json.loads(line)
        for line in (tmp_path / "headroom" / "headroom_per_cloud.jsonl")
        .read_text()
        .splitlines()
    ]

    expected_rows = 2 * 2 * (1 + 2 + 4 + 1)
    assert result["per_cloud_rows"] == expected_rows
    assert resumed["resumed_rows"] == expected_rows
    assert resumed["per_cloud_rows"] == expected_rows
    assert calls == first_call_count
    assert result["contract_checks"]["complete_factorial"]
    assert result["contract_checks"]["multi_start_selected_by_source_only"]
    assert {row["method"] for row in rows} == {
        "fps",
        "refiner",
        "adam_ste",
        "adam_multistart",
    }
    assert all(row["encode_seconds_cpu"] >= 0.0 for row in rows)
    assert all(row["encode_seconds_mps"] is None for row in rows)
    assert all(row["quality_device"] == "cpu" for row in rows)
    assert all(
        row["timing_columns_are_independent_device_reexecutions"] for row in rows
    )
    assert all(
        len(bytes.fromhex(row["stream_hex"])) == expected_stream_bytes(2, 8)
        for row in rows
    )
    assert all(
        decode_constellation(bytes.fromhex(row["stream_hex"])).stream_bytes == 20
        for row in rows
    )
