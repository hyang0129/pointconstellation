"""Focused tests for Experiment 032 constellation stability."""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.bitstream import encode_constellation
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.stability_analysis import (
    ConstellationStabilityConfig,
    constellation_matching_metrics,
    constellation_repeatability,
    evaluate_cross_decoder,
    pca_canonical_frame,
    random_rigid_transform,
    run_constellation_stability,
)
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    _decoder,
    _refiner,
)

FIXTURE_ROOT = Path("tests/fixtures/meshes").resolve()
FIXTURE_MANIFEST = Path("tests/fixtures/mesh_manifest.json")


def test_matching_metric_recovers_known_unordered_correspondences() -> None:
    bits = 8
    step = 2.0 / ((1 << bits) - 1)
    first = np.asarray(
        [
            [-0.75, -0.50, -0.25],
            [-0.25, 0.25, 0.50],
            [0.25, -0.25, 0.75],
            [0.75, 0.50, -0.50],
        ]
    )
    second = first[[2, 0, 3, 1]].copy()
    second[3, 0] += 3.0 * step

    metrics = constellation_matching_metrics(
        first,
        second,
        coordinate_bits=bits,
        radii_bins=(1, 3),
    )

    assert metrics["repeatability_by_radius_bins"]["1"] == pytest.approx(0.75)
    assert metrics["repeatability_by_radius_bins"]["3"] == pytest.approx(1.0)
    assert metrics["hausdorff_lattice_bins"] == pytest.approx(3.0)
    assert constellation_repeatability(
        first,
        second,
        coordinate_bits=bits,
        radius_bins=1,
    ) == pytest.approx(0.75)


def test_rigid_transform_bookkeeping_and_pca_alignment_are_consistent() -> None:
    rng = np.random.default_rng(37)
    points = rng.normal(size=(64, 3)) * np.asarray([0.35, 0.2, 0.08])
    transform = random_rigid_transform(
        points,
        seed=41,
        maximum_translation=0.1,
    )
    transformed = transform.apply(points)

    assert np.max(np.abs(transform.inverse(transformed) - points)) < 1e-12
    assert np.all(transformed >= -1.0) and np.all(transformed <= 1.0)
    assert np.linalg.det(transform.rotation) == pytest.approx(1.0)

    canonical = pca_canonical_frame(points).apply(points)
    transformed_canonical = pca_canonical_frame(transformed).apply(transformed)
    assert np.max(np.abs(canonical - transformed_canonical)) < 1e-10


class _FixtureDecoder(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.last_input: torch.Tensor | None = None

    def forward(
        self, constellation: torch.Tensor, *, num_output_points: int
    ) -> torch.Tensor:
        self.last_input = constellation.detach().clone()
        repeats = math.ceil(num_output_points / constellation.shape[1])
        output = constellation.repeat(1, repeats, 1)[:, :num_output_points]
        return output + self.offset


def test_cross_decoder_plumbing_transfers_only_serialized_coordinates() -> None:
    coordinates = np.asarray([[-0.5, 0.0, 0.5], [0.5, 0.0, -0.5]])
    stream = encode_constellation(
        coordinates,
        bits=8,
        mode="free",
        output_points=4,
    )
    source = np.repeat(coordinates, 2, axis=0)
    matched_decoder = _FixtureDecoder(0.0)
    cross_decoder = _FixtureDecoder(0.2)

    matched = evaluate_cross_decoder(
        matched_decoder,
        stream,
        source_points=source,
        distance_chunk_size=4,
    )
    crossed = evaluate_cross_decoder(
        cross_decoder,
        stream,
        source_points=source,
        distance_chunk_size=4,
    )

    assert matched_decoder.last_input is not None
    assert cross_decoder.last_input is not None
    assert torch.equal(matched_decoder.last_input, cross_decoder.last_input)
    assert matched_decoder.last_input.shape == (1, 2, 3)
    assert crossed.source_chamfer_mse > matched.source_chamfer_mse


def _fixture_stability_config(
    tmp_path: Path,
) -> tuple[Path, StabilityExperimentConfig]:
    manifest = json.loads(FIXTURE_MANIFEST.read_text())
    for split, records in manifest["splits"].items():
        official_split = "train" if split == "train" else "test"
        for record in records:
            record["official_split"] = official_split
    manifest_path = tmp_path / "mesh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    stability_path = tmp_path / "stability.json"
    stability = StabilityExperimentConfig(
        dataset_kind="mesh_manifest",
        dataset_root=str(FIXTURE_ROOT),
        dataset_manifest=str(manifest_path),
        calibration_split="train",
        mesh_ood_split="category_ood",
        num_points=8,
        constellation_size=2,
        training_constellation_sizes=(2, 4),
        coordinate_bits=8,
        train_samples=1,
        calibration_samples=1,
        validation_samples=2,
        ood_samples=2,
        batch_size=1,
        decoder_seeds=(1, 2),
        refiner_seeds=(3, 4),
        baseline_decoder_epochs=1,
        stabilized_decoder_epochs=2,
        feature_width=8,
        num_heads=2,
        recurrent_steps=1,
        distance_chunk_size=8,
        bootstrap_samples=100,
        output_dir=str(tmp_path / "unused"),
    )
    stability_path.write_text(json.dumps(asdict(stability), indent=2) + "\n")
    return stability_path, stability


def _write_fixture_artifact(
    tmp_path: Path, stability: StabilityExperimentConfig
) -> Path:
    artifact = tmp_path / "stability_artifact"
    artifact.mkdir()
    data_protocol = _data_protocol(stability, _datasets(stability))
    (artifact / "stability_metrics.json").write_text(
        json.dumps(
            {
                "config": asdict(stability),
                "contract_checks": {"fixture": True},
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
    return artifact


def test_tiny_fixture_runner_covers_repeatability_and_cross_decoder(
    tmp_path: Path,
) -> None:
    stability_path, stability = _fixture_stability_config(tmp_path)
    artifact = _write_fixture_artifact(tmp_path, stability)
    config = ConstellationStabilityConfig(
        stability_config=str(stability_path),
        stability_artifact_dir=str(artifact),
        decoder_seeds=(1, 2),
        refiner_seeds=(3,),
        splits=("validation",),
        max_clouds_per_split=1,
        repeatability_radii_bins=(1, 4),
        adam_evaluations=1,
        bootstrap_samples=100,
        gate_radius_bins=4,
        output_dir=str(tmp_path / "experiment_032"),
    )

    result = run_constellation_stability(config, device_name="cpu")

    assert result["comparison_rows"] == 2 * 4 * 4
    assert result["comparison_rows"] == result["expected_comparison_rows"]
    assert result["cross_decoder_rows"] == 2 * 4 * 2
    assert result["cross_decoder_rows"] == result["expected_cross_decoder_rows"]
    assert all(result["contract_checks"].values())
    assert result["statistics"]["gate_g_b3"]["applies_to"] == (
        "validation refiner factorial"
    )
    assert (tmp_path / "experiment_032" / "constellation_pairs.jsonl").is_file()
    assert (tmp_path / "experiment_032" / "cross_decoder.jsonl").is_file()
