# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.stability_experiment import (
    ARMS,
    StabilityExperimentConfig,
    _datasets,
    _decoder,
    _load_reused_decoders,
    _membership,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_SCRIPT = PROJECT_ROOT / "scripts" / "make_experiment_038_configs.py"
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_experiment_038.py"


def _load_run_script():
    spec = importlib.util.spec_from_file_location("run_experiment_038", RUN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_configs_are_deterministic_and_match_experiment_033(
    tmp_path: Path,
) -> None:
    output = tmp_path / "configs"
    command = [
        sys.executable,
        str(CONFIG_SCRIPT),
        "--output-config-dir",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    first = {path.name: path.read_bytes() for path in sorted(output.glob("*.json"))}
    subprocess.run(command, check=True, capture_output=True, text=True)
    repeated = {path.name: path.read_bytes() for path in sorted(output.glob("*.json"))}
    subprocess.run([*command, "--check"], check=True, capture_output=True, text=True)

    base = json.loads(
        (PROJECT_ROOT / "configs/experiment_019_stability_modelnet40.json").read_text()
    )
    objective = json.loads(
        (PROJECT_ROOT / "configs/experiment_033_objective_sweep.json").read_text()
    )
    missing = [
        regime for regime in objective["regimes"] if regime["name"] != "k8_n2048"
    ]

    assert first == repeated
    assert len(first) == len(missing) == 11
    assert set(first) == {Path(regime["stability_config"]).name for regime in missing}
    for regime in missing:
        generated = json.loads(first[Path(regime["stability_config"]).name].decode())
        assert generated["num_points"] == regime["num_points"]
        assert generated["constellation_size"] == regime["constellation_size"]
        assert generated["output_dir"] == regime["stability_artifact_dir"]
        for key in (
            "training_constellation_sizes",
            "decoder_seeds",
            "refiner_seeds",
            "baseline_decoder_epochs",
            "stabilized_decoder_epochs",
            "refiner_epochs",
            "coordinate_bits",
            "expected_manifest_sha256",
        ):
            assert generated[key] == base[key]
        if regime["num_points"] == 2048:
            assert generated["decoder_source_artifact_dir"] == base["output_dir"]
        else:
            assert "decoder_source_artifact_dir" not in generated


def test_runner_declares_supported_devices_and_one_seed_smoke(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "{cpu,mps,cuda}" in help_result.stdout

    module = _load_run_script()
    full = StabilityExperimentConfig.from_json(
        PROJECT_ROOT / "configs/experiment_038_stability_k4_n1024.json"
    )
    smoke = module._smoke_config(full, output_dir=tmp_path / "smoke")
    assert smoke.num_points == 1024
    assert smoke.constellation_size == 4
    assert (
        sum(
            (
                smoke.train_samples,
                smoke.calibration_samples,
                smoke.validation_samples,
                smoke.ood_samples,
            )
        )
        == 8
    )
    assert len(smoke.decoder_seeds) == len(smoke.refiner_seeds) == 1
    assert smoke.allow_single_seed
    assert smoke.coordinate_bits == 12


def test_n2048_decoder_reuse_verifies_hashes_and_training_curriculum(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source = StabilityExperimentConfig(
        num_points=16,
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
        reference_feature_dir=None,
        bootstrap_samples=100,
        output_dir=str(source_dir),
    )
    target = replace(
        source,
        constellation_size=4,
        decoder_source_artifact_dir=str(source_dir),
        output_dir=str(tmp_path / "target"),
    )
    datasets = _datasets(target)
    calibration_hash = _membership(datasets["calibration"])["sha256"]
    decoder = _decoder(source, torch.device("cpu"))
    state_hash = _state_hash(decoder)
    selection_record = {
        "epoch": 2,
        "kind": "ema",
        "calibration_chamfer_rmse": 0.25,
        "selected_state_hash": state_hash,
    }
    arms = {}
    decoder_dir = source_dir / "decoders" / "seed_1"
    decoder_dir.mkdir(parents=True)
    for arm in ARMS:
        checkpoint = decoder_dir / f"{arm}.pt"
        torch.save(
            {
                "model": decoder.state_dict(),
                "decoder_seed": 1,
                "arm": arm,
                "state_hash": state_hash,
                "selection": selection_record,
            },
            checkpoint,
        )
        arms[arm] = {
            "state_hash": state_hash,
            "checkpoint": str(checkpoint),
            "selection": selection_record,
        }
    (decoder_dir / "selection.json").write_text(
        json.dumps(
            {
                "decoder_seed": 1,
                "calibration_partition_sha256": calibration_hash,
                "arms": arms,
            }
        )
    )
    (source_dir / "stability_metrics.json").write_text(
        json.dumps(
            {
                "config": asdict(source),
                "factorial": {"complete": True},
                "contract_checks": {"fixture_contract": True},
                "decoder_records": [
                    {
                        "decoder_seed": 1,
                        "history": [],
                        "ema_decay": source.ema_decay,
                        "stabilized_candidates": [],
                        "arms": arms,
                    }
                ],
            }
        )
    )

    states, record = _load_reused_decoders(
        target,
        datasets,
        decoder_seed=1,
        device=torch.device("cpu"),
    )

    assert set(states) == set(ARMS)
    assert record["reuse"]["decoder_training_skipped"]
    assert record["reuse"]["source_constellation_size"] == 2
    assert record["reuse"]["requested_constellation_size"] == 4
    assert all(record["arms"][arm]["state_hash"] == state_hash for arm in ARMS)
