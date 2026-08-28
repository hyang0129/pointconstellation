"""Focused tests for Experiment 030 downstream classification."""

# ruff: noqa: E402

from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pointconstellation.codecs import GpccResult, Tmc3RatePoint
from pointconstellation.downstream_classification import (
    CachedRepresentation,
    ClassifierConfig,
    DownstreamClassificationConfig,
    DownstreamGpccConfig,
    FrozenRepresentationExtractor,
    RepresentationSpec,
    aggregate_accuracy,
    train_classifier,
)
from pointconstellation.stability_experiment import StabilityExperimentConfig


class _TinyDecoder(torch.nn.Module):
    def forward(
        self, constellation: torch.Tensor, *, num_output_points: int
    ) -> torch.Tensor:
        repeats = (
            num_output_points + constellation.shape[1] - 1
        ) // constellation.shape[1]
        return constellation.repeat(1, repeats, 1)[:, :num_output_points]


def test_encoder_api_excludes_labels_and_round_trips_exact_bytes() -> None:
    parameters = set(
        inspect.signature(FrozenRepresentationExtractor.extract).parameters
    )
    assert parameters == {"self", "source_points"}
    assert parameters.isdisjoint({"category", "categories", "label", "labels"})

    config = DownstreamClassificationConfig(
        constellation_sizes=(2,),
        feature_latent_dims=(),
        primary_constellation_size=2,
        include_adam=False,
        include_refiner=False,
        include_feature=False,
        include_fps=True,
        include_source=True,
        classifier_seeds=(1,),
        bootstrap_samples=100,
        official_stability_artifact_dir=None,
    )
    stability = StabilityExperimentConfig(
        num_points=8,
        constellation_size=2,
        training_constellation_sizes=(2,),
        coordinate_bits=8,
        train_samples=2,
        calibration_samples=1,
        validation_samples=1,
        ood_samples=1,
        batch_size=1,
        decoder_seeds=(1, 2),
        refiner_seeds=(3, 4),
        baseline_decoder_epochs=1,
        stabilized_decoder_epochs=2,
    )
    extractor = FrozenRepresentationExtractor(
        config,
        stability,
        decoder=_TinyDecoder().eval(),
        refiner=None,
        feature_codec=None,
        device=torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(17)
    source = 1.8 * torch.rand((2, 8, 3), generator=generator) - 0.9

    result = extractor.extract(source)

    assert set(result) == {"fps_raw_k0002", "fps_decoded_k0002", "source_full"}
    assert result["fps_raw_k0002"].stream_bytes.tolist() == [20.0, 20.0]
    assert all(len(value) == 64 for value in result["fps_raw_k0002"].stream_sha256)


def _fixture_cache(
    values: np.ndarray, categories: list[str], *, key: str
) -> CachedRepresentation:
    count, points, _ = values.shape
    return CachedRepresentation(
        spec=RepresentationSpec(
            key=key,
            method="fixture",
            view="raw",
            input_kind="point_set",
            representation_class="fixture",
            constellation_size=points,
            target_stream_bytes=None,
        ),
        values=values.astype(np.float32),
        lengths=np.full(count, points, dtype=np.int64),
        categories=np.asarray(categories, dtype=np.str_),
        model_ids=np.asarray([f"model_{index}" for index in range(count)]),
        sample_ids=np.arange(count, dtype=np.int64),
        stream_bytes=np.full(count, np.nan),
        stream_sha256=np.asarray([""] * count),
        rate_points=np.asarray([""] * count),
    )


def test_pointnet_classifier_learns_tiny_frozen_fixture() -> None:
    rng = np.random.default_rng(23)

    def clouds(center: float, count: int) -> np.ndarray:
        return rng.normal(center, 0.03, size=(count, 8, 3)).astype(np.float32)

    train_values = np.concatenate((clouds(-0.7, 16), clouds(0.7, 16)))
    validation_values = np.concatenate((clouds(-0.7, 8), clouds(0.7, 8)))
    train = _fixture_cache(
        train_values,
        ["negative"] * 16 + ["positive"] * 16,
        key="train",
    )
    validation = _fixture_cache(
        validation_values,
        ["negative"] * 8 + ["positive"] * 8,
        key="validation",
    )

    _, result = train_classifier(
        train,
        {"validation": validation},
        category_to_index={"negative": 0, "positive": 1},
        config=ClassifierConfig(
            epochs=15,
            batch_size=8,
            learning_rate=0.01,
            hidden_width=16,
            embedding_dim=8,
        ),
        seed=29,
        retrieval_k=5,
        device=torch.device("cpu"),
    )

    evaluation = result["splits"]["validation"]
    assert result["optimizer_updates"] == 60
    assert float(evaluation["correct"].mean()) == 1.0
    assert float(np.nanmean(evaluation["retrieval_ap"])) == 1.0


def test_accuracy_aggregation_reports_seed_and_cloud_ci() -> None:
    perfect = np.ones((3, 12), dtype=np.float64)

    first = aggregate_accuracy(
        perfect,
        samples=200,
        confidence_level=0.95,
        seed=31,
    )
    repeated = aggregate_accuracy(
        perfect,
        samples=200,
        confidence_level=0.95,
        seed=31,
    )

    assert first == repeated
    assert first["mean"] == 1.0
    assert first["per_seed"] == [1.0, 1.0, 1.0]
    assert first["confidence_interval_lower"] == 1.0
    assert first["confidence_interval_upper"] == 1.0


def test_gpcc_control_selects_nearest_actual_bytes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpcc = DownstreamGpccConfig(
        executable="unused",
        rate_points=(
            Tmc3RatePoint("low", ("--positionQuantizationScale=0.1",)),
            Tmc3RatePoint("high", ("--positionQuantizationScale=0.2",)),
        ),
        position_bits=8,
    )
    config = DownstreamClassificationConfig(
        constellation_sizes=(2, 4),
        feature_latent_dims=(),
        primary_constellation_size=2,
        include_adam=False,
        include_refiner=False,
        include_feature=False,
        include_fps=False,
        include_source=False,
        classifier_seeds=(1,),
        bootstrap_samples=100,
        official_stability_artifact_dir=None,
        gpcc=gpcc,
    )
    stability = StabilityExperimentConfig(
        num_points=8,
        constellation_size=2,
        training_constellation_sizes=(2, 4),
        coordinate_bits=8,
        train_samples=2,
        calibration_samples=1,
        validation_samples=1,
        ood_samples=1,
        batch_size=1,
        decoder_seeds=(1, 2),
        refiner_seeds=(3, 4),
        baseline_decoder_epochs=1,
        stabilized_decoder_epochs=2,
    )

    def fake_tmc3(*args, **kwargs) -> GpccResult:
        rate_point = kwargs["rate_point"]
        work_dir = kwargs["work_dir"]
        work_dir.mkdir(parents=True, exist_ok=True)
        size = 20 if rate_point.name == "low" else 30
        (work_dir / "stream.bin").write_bytes(bytes(size))
        points = 3 if rate_point.name == "low" else 5
        return GpccResult(
            reconstruction=np.zeros((points, 3), dtype=np.float32),
            stream_bytes=size,
            encode_seconds=0.0,
            decode_seconds=0.0,
            encoder_command=("fixture",),
            decoder_command=("fixture",),
            encoder_stdout="",
            decoder_stdout="",
        )

    monkeypatch.setattr(
        "pointconstellation.downstream_classification.run_tmc3", fake_tmc3
    )
    extractor = FrozenRepresentationExtractor(
        config,
        stability,
        decoder=_TinyDecoder().eval(),
        refiner=None,
        feature_codec=None,
        device=torch.device("cpu"),
    )

    result = extractor.extract_gpcc(
        torch.zeros((1, 8, 3)),
        work_root=tmp_path,
        sample_offset=0,
    )

    assert result["gpcc_decoded_k0002"].stream_bytes.tolist() == [20.0]
    assert result["gpcc_decoded_k0002"].rate_points == ("low",)
    assert result["gpcc_decoded_k0004"].stream_bytes.tolist() == [30.0]
    assert result["gpcc_decoded_k0004"].rate_points == ("high",)
