# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _feature_reference_comparison,
    _serialized_coordinates,
    _two_way_components,
)


def test_serialized_coordinates_return_header_payload_packets() -> None:
    config = StabilityExperimentConfig(
        num_points=16,
        constellation_size=2,
        training_constellation_sizes=(2, 4),
        coordinate_bits=7,
    )
    coordinates = torch.zeros((2, 2, 3), dtype=torch.float32)

    _, packets, exact, lattice_exact = _serialized_coordinates(
        coordinates, config=config, mode="free"
    )

    assert exact and lattice_exact
    assert len(packets) == 2
    assert all(
        packet.header_bytes + packet.payload_bytes == packet.stream_bytes
        for packet in packets
    )


def test_config_rejects_test_calibration_and_incomplete_rate_curriculum() -> None:
    with pytest.raises(ValueError, match="train or calibration"):
        StabilityExperimentConfig(calibration_split="validation")
    with pytest.raises(ValueError, match="include K"):
        StabilityExperimentConfig(training_constellation_sizes=(4, 16, 32))
    with pytest.raises(ValueError, match="bootstrap_samples"):
        StabilityExperimentConfig(bootstrap_samples=99)


def test_procedural_partitions_are_disjoint_and_deterministic() -> None:
    config = StabilityExperimentConfig(
        num_points=16,
        constellation_size=2,
        training_constellation_sizes=(2, 4),
        train_samples=4,
        calibration_samples=2,
        validation_samples=2,
        ood_samples=2,
        batch_size=2,
        decoder_seeds=(1, 2),
        refiner_seeds=(3, 4),
        baseline_decoder_epochs=1,
        stabilized_decoder_epochs=2,
    )

    first = _data_protocol(config, _datasets(config))
    repeated = _data_protocol(config, _datasets(config))

    assert first == repeated
    assert first["all_partitions_pairwise_disjoint"]
    assert first["partitions"]["train"]["count"] == 4
    assert first["partitions"]["calibration"]["count"] == 2


def test_two_way_components_identify_decoder_dominance() -> None:
    result = _two_way_components(
        np.asarray([[0.0, 0.01, -0.01], [1.0, 1.01, 0.99]]),
        decoder_seeds=(7, 17),
        refiner_seeds=(101, 211, 307),
    )

    assert result["variance_fraction"]["decoder"] > 0.99
    assert "unconstrained_variance" in result
    assert "nonnegative_constrained_variance" in result


def _feature_row(
    *, split: str, family: str, model_id: str, mse: float
) -> dict[str, object]:
    return {
        "split": split,
        "family": family,
        "model_id": model_id,
        "method": "feature_latent",
        "constellation_size": 2,
        "stream_bytes": 20,
        "chamfer_mse": mse,
        "fresh_chamfer_mse": mse * 1.1,
    }


def test_feature_reference_resamples_model_factors_independently(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "feature"
    reference.mkdir()
    (reference / "multiseed_metrics.json").write_text(
        json.dumps(
            {
                "config": {
                    "data_seed": 19,
                    "num_points": 16,
                    "primary_constellation_size": 2,
                    "coordinate_bits": 8,
                }
            }
        )
    )
    clouds = (("alpha", "a"), ("beta", "b"))
    for seed in (7, 17, 29):
        seed_dir = reference / f"seed_{seed}"
        seed_dir.mkdir()
        feature_rows = [
            _feature_row(split=split, family=family, model_id=model_id, mse=0.04)
            for split in ("validation", "category_ood")
            for family, model_id in clouds
        ]
        (seed_dir / "per_cloud.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in feature_rows)
        )

    coordinate_rows = []
    for decoder_seed in (1, 2):
        for refiner_seed in (3, 4):
            for split in ("validation", "ood"):
                for family, model_id in clouds:
                    coordinate_rows.append(
                        {
                            "arm": "stabilized",
                            "split": split,
                            "method": "refiner",
                            "decoder_seed": decoder_seed,
                            "refiner_seed": refiner_seed,
                            "family": family,
                            "model_id": model_id,
                            "chamfer_mse": 0.01,
                            "fresh_chamfer_mse": 0.011,
                        }
                    )
    config = StabilityExperimentConfig(
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
        data_seed=19,
        baseline_decoder_epochs=1,
        stabilized_decoder_epochs=2,
        reference_feature_dir=str(reference),
        bootstrap_samples=100,
    )

    result = _feature_reference_comparison(coordinate_rows, config)

    assert result["status"] == "complete"
    assert result["feature_seeds"] == [7, 17, 29]
    assert result["primary"]["coordinate_decoder_seeds"] == 2
    assert result["primary"]["coordinate_refiner_seeds"] == 2
    assert result["primary"]["feature_codec_seeds"] == 3
    assert result["primary"]["confidence_interval_lower_percent"] > 0
    assert result["learned_codec_gate_passes"]
