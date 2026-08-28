"""Focused tests for Experiment 025's stabilized Adam rate sweep."""

# ruff: noqa: E402

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.bitstream import (
    HEADER,
    expected_payload_bytes,
    expected_stream_bytes,
)
from pointconstellation.codecs import OfficialMetricResult
from pointconstellation.headroom_experiment import _metadata
from pointconstellation.rate_sweep_experiment import (
    RateSweepExperimentConfig,
    _rate_fields,
    run_rate_sweep_experiment,
)
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _decoder,
    _refiner,
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


def _write_gpcc_reference(path: Path, stability: StabilityExperimentConfig) -> None:
    rows = []
    validation = _datasets(stability)["validation"]
    for sample_index in range(2):
        metadata = _metadata(validation[sample_index], sample_index)
        for rate_point, payload_bytes in (("low", 5), ("high", 20)):
            rows.append(
                {
                    "split": "validation",
                    **metadata,
                    "method": "gpcc_octree",
                    "rate_point": rate_point,
                    "header_bytes": 10,
                    "payload_bytes": payload_bytes,
                    "stream_bytes": payload_bytes + 10,
                    "payload_bpp": 8.0 * payload_bytes / stability.num_points,
                    "actual_stream_bpp": 8.0
                    * (payload_bytes + 10)
                    / stability.num_points,
                    "official_d1_mse": 2.0 + sample_index,
                    "official_d2_mse": 1.0 + sample_index,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_rate_fields_match_exact_bitstream_sizes() -> None:
    for constellation_size in (4, 6, 8, 12, 16):
        for bits in (8, 10, 12):
            rate = _rate_fields(constellation_size, bits, 2048)

            assert rate["header_bytes"] == HEADER.size
            assert rate["payload_bytes"] == expected_payload_bytes(
                constellation_size, bits
            )
            assert rate["stream_bytes"] == expected_stream_bytes(
                constellation_size, bits
            )
            assert rate["header_bytes"] + rate["payload_bytes"] == rate["stream_bytes"]


def test_tiny_rate_sweep_runs_end_to_end_and_resumes_without_duplicates(
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
    gpcc_path = tmp_path / "gpcc_per_cloud.jsonl"
    _write_gpcc_reference(gpcc_path, stability)
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
        "pointconstellation.rate_sweep_experiment.run_pc_error", fake_pc_error
    )
    config = RateSweepExperimentConfig(
        stability_config=str(stability_path),
        stability_artifact_dir=str(tmp_path / "stability"),
        pc_error_executable=str(executable),
        gpcc_reference_path=str(gpcc_path),
        position_bits=8,
        constellation_sizes=(2, 4),
        coordinate_bits=(8,),
        decoder_seeds=(1,),
        refiner_seeds=(3,),
        refiner_constellation_size=2,
        random_start_seeds=(5, 7),
        adam_evaluations=2,
        splits=("validation",),
        max_clouds_per_split=2,
        batch_size=2,
        bootstrap_samples=100,
        output_dir=str(tmp_path / "rate_sweep"),
    )

    result = run_rate_sweep_experiment(config, device_name="cpu")
    first_call_count = calls
    resumed = run_rate_sweep_experiment(config, device_name="cpu")
    cell_paths = sorted(
        (tmp_path / "rate_sweep" / "cells").glob("*/rate_sweep_per_cloud.jsonl")
    )
    rows = [
        json.loads(line)
        for path in cell_paths
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    assert result["completed_cells"] == result["expected_cells"] == 2
    assert result["per_cloud_rows"] == 14
    assert resumed["resumed_rows"] == 14
    assert resumed["per_cloud_rows"] == 14
    assert calls == first_call_count
    assert len(rows) == 14
    assert len(
        {
            (
                row["constellation_size"],
                row["coordinate_bits"],
                row["split"],
                row["method"],
                row["decoder_seed"],
                row["refiner_seed"],
                row["sample_id"],
            )
            for row in rows
        }
    ) == len(rows)
    assert result["contract_checks"]["complete_grid"]
    assert all(
        row["stream_bytes"]
        == expected_stream_bytes(row["constellation_size"], row["coordinate_bits"])
        for row in rows
    )
    assert all(
        row["payload_bytes"]
        == expected_payload_bytes(row["constellation_size"], row["coordinate_bits"])
        for row in rows
    )
    assert all(row["d1_mse"] >= 0 and row["d2_mse"] >= 0 for row in rows)
    assert {comparison["comparison_role"] for comparison in result["comparisons"]} == {
        "method_vs_fps",
        "method_vs_nearest_gpcc_payload_rate",
    }
    assert (tmp_path / "rate_sweep" / "rate_sweep_curve.json").is_file()
    assert (tmp_path / "rate_sweep" / "rate_sweep_table.md").is_file()
