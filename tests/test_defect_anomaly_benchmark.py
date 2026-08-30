"""Focused tests for the Experiment 041 anomaly benchmark."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pointconstellation.codecs import GpccResult, GpccStreamBreakdown
from pointconstellation.defect_anomaly_benchmark import (
    BenchmarkCloud,
    CodecArm,
    DefectAnomalyBenchmarkConfig,
    KNNNormalManifoldScorer,
    KNNScorerConfig,
    PointCloudCodec,
    _decode_samples,
    _transfer_point_labels_device,
    assert_matched_bytes,
    binary_auprc,
    binary_auroc,
    build_diagnostic_subset_codec_arms,
    load_external_anomaly_manifest,
    run_defect_anomaly_benchmark,
)
from pointconstellation.defects import DEFECT_TYPES, inject_defect
from pointconstellation.selective_experiment import SelectiveExperimentConfig


def _sphere(count: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(71)
    z = rng.uniform(-1.0, 1.0, count)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = np.sqrt(1.0 - z * z)
    normals = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    return (0.7 * normals).astype(np.float32), normals.astype(np.float32)


def _reference_nearest_squared(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.empty(len(query), dtype=np.float64)
    indices = np.empty(len(query), dtype=np.int64)
    reference_norms = np.einsum("ij,ij->i", reference, reference)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        rows = query[start:stop]
        squared = (
            np.einsum("ij,ij->i", rows, rows)[:, None]
            + reference_norms[None]
            - 2.0 * rows @ reference.T
        )
        np.maximum(squared, 0.0, out=squared)
        selected = np.argmin(squared, axis=1)
        nearest = np.partition(squared, neighbors - 1, axis=1)[:, :neighbors]
        distances[start:stop] = nearest.mean(axis=1)
        indices[start:stop] = selected
    return distances, indices


def _reference_score_points(
    scorer: KNNNormalManifoldScorer, points: np.ndarray
) -> np.ndarray:
    query = np.asarray(points, dtype=np.float32)
    query64 = query.astype(np.float64, copy=False)
    candidate_scores = []
    for candidate in scorer._candidate_indices(query):
        reference = scorer._references[int(candidate)].astype(
            np.float64, copy=False
        )
        forward, _ = _reference_nearest_squared(
            query64,
            reference,
            chunk_size=scorer.config.distance_chunk_size,
            neighbors=min(scorer.config.neighbors, len(reference)),
        )
        reverse, assigned = _reference_nearest_squared(
            reference,
            query64,
            chunk_size=scorer.config.distance_chunk_size,
            neighbors=min(scorer.config.neighbors, len(query64)),
        )
        attributed = np.zeros(len(query), dtype=np.float64)
        np.maximum.at(attributed, assigned, reverse)
        candidate_scores.append(np.sqrt(np.maximum(forward, attributed)))
    return np.min(np.stack(candidate_scores), axis=0)


def _reference_binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _reference_binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    true_positives = 0
    predicted = 0
    weighted_precision = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        group_positives = int(labels[order[start:stop]].sum())
        true_positives += group_positives
        predicted += stop - start
        weighted_precision += group_positives * true_positives / predicted
        start = stop
    return weighted_precision / positives


def _reference_transfer_labels(
    reference: np.ndarray, labels: np.ndarray, query: np.ndarray
) -> np.ndarray:
    reference64 = reference.astype(np.float64)
    query64 = query.astype(np.float64)
    canonical = np.lexsort(
        (reference64[:, 2], reference64[:, 1], reference64[:, 0])
    )
    canonical_ranks = np.empty(len(reference), dtype=np.int64)
    canonical_ranks[canonical] = np.arange(len(reference))
    squared = np.sum((query64[:, None] - reference64[None]) ** 2, axis=2)
    nearest = [np.lexsort((canonical_ranks, row))[0] for row in squared]
    return labels[np.asarray(nearest)]


def _near_boundary_planes(count: int) -> tuple[np.ndarray, np.ndarray]:
    if count % 2:
        raise ValueError("boundary fixture count must be even")
    rng = np.random.default_rng(79)
    first = rng.uniform(-0.25, 0.25, size=count // 2)
    second = rng.uniform(-0.25, 0.25, size=count // 2)
    positive = np.column_stack(
        (np.full(len(first), 0.995), first, second)
    )
    negative = positive.copy()
    negative[:, 0] = -0.995
    points = np.concatenate((positive, negative)).astype(np.float32)
    normals = np.zeros_like(points)
    normals[:, 0] = np.sign(points[:, 0])
    return points, normals


def _install_fake_real_provider_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> type:
    class MaximumEnforcingDecoderContext:
        instances = []
        search_calls = 0

        def __init__(self, config, *, decoder_seed, device_name):
            del decoder_seed, device_name
            self.config = config
            self.stability = SimpleNamespace(num_points=2048)
            self.requested_output_points = []
            self.__class__.instances.append(self)

        def search(self, source, *, constellation_size, bits, output_points):
            del bits
            self.__class__.search_calls += 1
            if output_points > self.stability.num_points:
                raise ValueError("num_output_points exceeds the configured maximum")
            self.requested_output_points.append(output_points)
            return np.asarray(source[:constellation_size], dtype=np.float64)

        def decode(self, coordinates, output_points):
            if output_points > self.stability.num_points:
                raise ValueError("num_output_points exceeds the configured maximum")
            self.requested_output_points.append(output_points)
            repeats = int(np.ceil(output_points / len(coordinates)))
            return np.tile(coordinates, (repeats, 1))[:output_points]

    class FakeGpccFrontier:
        def __init__(self, config, experiment_040):
            del config, experiment_040
            self.sources = {}
            self.metadata_rows = {}

        def encode(self, source, *, payload_budget, target_bytes):
            stream = payload_budget.to_bytes(2, "big") + bytes(target_bytes + 1)
            self.sources[stream] = np.asarray(source, dtype=np.float32).copy()
            payload_bytes = payload_budget + 1
            self.metadata_rows[(stream, payload_budget)] = {
                "codec_header_bytes": len(stream) - payload_bytes,
                "codec_payload_bytes": payload_bytes,
                "payload_byte_delta": 1,
                "complete_stream_byte_delta": len(stream) - target_bytes,
                "codec_rate_point": "fake_nearest",
            }
            return stream

        def decode(self, stream):
            return self.sources[stream].copy()

        def metadata(self, stream, *, payload_budget):
            return self.metadata_rows[(stream, payload_budget)]

    monkeypatch.setattr(
        "pointconstellation.exp040_defect_codecs.FrozenExperiment040CodecContext",
        MaximumEnforcingDecoderContext,
    )
    monkeypatch.setattr(
        "pointconstellation.exp040_defect_codecs._GpccFrontier", FakeGpccFrontier
    )
    return MaximumEnforcingDecoderContext


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


def test_optimized_scorer_transfer_and_metrics_match_reference_to_float64() -> None:
    rng = np.random.default_rng(411_041)
    references = [
        rng.normal(size=(97, 3)).astype(np.float32) for _ in range(7)
    ]
    # Duplicates exercise exact distance ties in top-k and reverse attribution.
    references[2][5] = references[2][1]
    queries = [rng.normal(size=(83, 3)).astype(np.float32) for _ in range(4)]
    queries[1][8] = queries[1][3]
    scorer = KNNNormalManifoldScorer(
        KNNScorerConfig(
            neighbors=3,
            points_per_reference=61,
            maximum_reference_clouds=7,
            cloud_candidates=3,
            distance_chunk_size=17,
        ),
        seed=41,
        device_name="cpu",
    ).fit(references)

    optimized_scores = scorer.score_points_batch(queries)
    for query, optimized in zip(queries, optimized_scores, strict=True):
        reference = _reference_score_points(scorer, query)
        np.testing.assert_allclose(optimized, reference, rtol=0.0, atol=1e-9)

    metric_cases = [
        (
            rng.integers(0, 2, size=257, dtype=np.uint8),
            np.round(rng.normal(size=257), decimals=1),
        ),
        (np.zeros(19, dtype=np.uint8), np.ones(19)),
        (np.ones(19, dtype=np.uint8), np.ones(19)),
        (np.asarray([0, 1, 0, 1], dtype=np.uint8), np.zeros(4)),
    ]
    for labels, scores in metric_cases:
        optimized_auroc = binary_auroc(labels, scores)
        optimized_auprc = binary_auprc(labels, scores)
        reference_auroc = _reference_binary_auroc(labels, scores)
        reference_auprc = _reference_binary_auprc(labels, scores)
        if np.isnan(reference_auroc):
            assert np.isnan(optimized_auroc)
        else:
            assert optimized_auroc == pytest.approx(reference_auroc, abs=1e-9)
        if np.isnan(reference_auprc):
            assert np.isnan(optimized_auprc)
        else:
            assert optimized_auprc == pytest.approx(reference_auprc, abs=1e-9)

    tie_reference = np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    tie_labels = np.asarray([0, 1, 0], dtype=np.uint8)
    tie_query = np.asarray(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32
    )
    optimized_labels = _transfer_point_labels_device(
        tie_reference,
        tie_labels,
        tie_query,
        device=torch.device("cpu"),
        chunk_size=1,
    )
    assert np.array_equal(
        optimized_labels,
        _reference_transfer_labels(tie_reference, tie_labels, tie_query),
    )


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
    config = replace(
        config, codec_provider=None, diagnostic_subset_codecs=True
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

    assert not smoke.diagnostic_subset_codecs
    assert not full.diagnostic_subset_codecs
    assert smoke.codec_provider == full.codec_provider == (
        "pointconstellation.exp040_defect_codecs:build_codec_arms"
    )
    assert smoke.experiment_040_decoder_seed == 7
    assert smoke.selective_score_method == "decoder_residual"
    assert smoke.selective_preserved_fraction == 0.5
    assert smoke.num_points == full.num_points == 2048
    assert smoke.defect_types == ("bump", "thin_spur")
    assert len(smoke.scorer_seeds) == len(full.scorer_seeds) == 3
    assert smoke.payload_budgets == full.payload_budgets == (40, 52, 64, 78, 96, 110)


def test_real_provider_arms_declare_consistent_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pointconstellation.exp040_defect_codecs import build_codec_arms

    _install_fake_real_provider_dependencies(monkeypatch)
    config = DefectAnomalyBenchmarkConfig.from_json(
        Path("configs/experiment_041_defect_anomaly_smoke.json")
    )
    config = replace(config, output_dir=str(tmp_path / "output"))
    arms = build_codec_arms(config=config, device_name="cpu")
    points, _ = _sphere(config.num_points)

    assert len(arms) == 4 * len(config.payload_budgets)
    for arm in arms:
        stream = arm.codec.encode(points)
        metadata = arm.codec.rate_metadata(stream)
        assert metadata["codec_header_bytes"] + metadata["codec_payload_bytes"] == len(
            stream
        )
        assert metadata["payload_byte_delta"] == (
            metadata["codec_payload_bytes"] - arm.payload_budget_bytes
        )
        assert metadata["complete_stream_byte_delta"] == len(stream) - arm.target_bytes
        if arm.name != config.gpcc_arm:
            assert len(stream) == arm.target_bytes
        assert arm.codec.decode(stream).shape[1] == 3


def test_adam_ste_search_cache_is_hashed_reused_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pointconstellation.exp040_defect_codecs import build_codec_arms

    decoder_context = _install_fake_real_provider_dependencies(monkeypatch)
    config = DefectAnomalyBenchmarkConfig.from_json(
        Path("configs/experiment_041_defect_anomaly_smoke.json")
    )
    config = replace(
        config,
        output_dir=str(tmp_path / "output"),
        payload_budgets=(40,),
        primary_payload_budget=40,
    )
    points, _ = _sphere(config.num_points)

    first = build_codec_arms(config=config, device_name="cpu")[0]
    first_stream = first.codec.encode(points)
    assert decoder_context.search_calls == 1
    cache_manifests = list(
        (Path(config.output_dir) / "codec_scratch" / "adam_ste").rglob(
            "cache_manifest.json"
        )
    )
    assert len(cache_manifests) == 1

    second = build_codec_arms(config=config, device_name="cpu")[0]
    assert second.codec.encode(points) == first_stream
    assert decoder_context.search_calls == 1

    coordinates = cache_manifests[0].with_name("coordinates.npy")
    coordinates.write_bytes(b"invalidated cache")
    third = build_codec_arms(config=config, device_name="cpu")[0]
    assert third.codec.encode(points) == first_stream
    assert decoder_context.search_calls == 2


@pytest.mark.parametrize("num_points", (2048, 1024))
def test_every_defect_preserves_regime_through_real_provider_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_points: int
) -> None:
    from pointconstellation.exp040_defect_codecs import build_codec_arms

    decoder_context = _install_fake_real_provider_dependencies(monkeypatch)
    config = DefectAnomalyBenchmarkConfig.from_json(
        Path("configs/experiment_041_defect_anomaly_smoke.json")
    )
    config = replace(
        config,
        num_points=num_points,
        output_dir=str(tmp_path / "output"),
        payload_budgets=(40,),
        primary_payload_budget=40,
    )
    provider_arms = build_codec_arms(config=config, device_name="cpu")

    class CapturingCodec:
        def __init__(self, codec):
            self.codec = codec
            self.sources = []

        def encode(self, source):
            self.sources.append(np.asarray(source, dtype=np.float32).copy())
            return self.codec.encode(source)

        def decode(self, stream):
            return self.codec.decode(stream)

        def rate_metadata(self, stream):
            return self.codec.rate_metadata(stream)

    captures = []
    arms = []
    for arm in provider_arms:
        capture = CapturingCodec(arm.codec)
        captures.append(capture)
        arms.append(replace(arm, codec=capture))

    points, normals = _near_boundary_planes(num_points)
    samples = []
    injected = []
    for index, defect_type in enumerate(DEFECT_TYPES):
        defect = inject_defect(
            points,
            defect_type,
            seed=100 + index,
            fraction=0.03,
            normals=normals,
        )
        injected.append(defect)
        samples.append(
            BenchmarkCloud(
                split="validation",
                cloud_id=f"boundary:{defect_type}",
                category="boundary_fixture",
                defect_type=defect_type,
                size_stratum="medium_2_4pct",
                declared_fraction=defect.declared_fraction,
                points=defect.points,
                point_labels=defect.point_labels,
                cloud_label=1,
                removed_count=defect.removed_count,
                domain_scale_factor=defect.domain_scale_factor,
            )
        )

    decoded, checks = _decode_samples(
        samples, arms, expected_point_count=num_points
    )

    assert all(checks.values())
    assert all(
        len(result.points) == num_points
        and len(result.point_labels) == num_points
        and np.all(result.points >= -1.0)
        and np.all(result.points <= 1.0)
        for result in injected
    )
    assert any(result.domain_scale_factor < 1.0 for result in injected)
    assert decoder_context.instances
    assert all(
        request == num_points
        for context in decoder_context.instances
        for request in context.requested_output_points
    )
    for capture in captures:
        assert len(capture.sources) == len(samples)
        for source, sample in zip(capture.sources, samples, strict=True):
            assert np.array_equal(source, sample.points)
    raw_rows = [row for row in decoded if row.arm == "raw"]
    assert len(raw_rows) == len(samples)
    for row, sample in zip(raw_rows, samples, strict=True):
        assert np.array_equal(row.decoded_points, sample.points)
        assert np.array_equal(row.decoded_labels, sample.point_labels)


def test_real_provider_rejects_source_outside_declared_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pointconstellation.exp040_defect_codecs import build_codec_arms

    _install_fake_real_provider_dependencies(monkeypatch)
    config = DefectAnomalyBenchmarkConfig.from_json(
        Path("configs/experiment_041_defect_anomaly_smoke.json")
    )
    config = replace(
        config,
        output_dir=str(tmp_path / "output"),
        payload_budgets=(40,),
        primary_payload_budget=40,
    )
    arm = build_codec_arms(config=config, device_name="cpu")[0]
    points, _ = _sphere(config.num_points + 1)

    with pytest.raises(ValueError, match="declared Experiment 041 regime"):
        arm.codec.encode(points)


def test_real_provider_rejects_regime_above_sealed_decoder_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pointconstellation.exp040_defect_codecs import build_codec_arms

    _install_fake_real_provider_dependencies(monkeypatch)
    config = DefectAnomalyBenchmarkConfig.from_json(
        Path("configs/experiment_041_defect_anomaly_smoke.json")
    )
    config = replace(config, num_points=2049)

    with pytest.raises(ValueError, match="sealed decoder maximum"):
        build_codec_arms(config=config, device_name="cpu")


def test_gpcc_provider_fresh_frontier_matches_payload_and_declares_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pointconstellation.exp040_defect_codecs import _GpccFrontier

    reference = tmp_path / "gpcc_reference.jsonl"
    reference.write_text(
        "\n".join(
            json.dumps(
                {
                    "method": "gpcc_octree",
                    "rate_point": name,
                    "encoder_args": [f"--positionQuantizationScale={scale}"],
                }
            )
            for name, scale in (("payload_35", 0.5), ("payload_70", 0.25))
        )
        + "\n"
    )
    calls = []

    def fake_run_tmc3(executable, source, *, rate_point, work_dir, **kwargs):
        del executable, kwargs
        calls.append(rate_point.name)
        payload_bytes = int(rate_point.name.removeprefix("payload_"))
        breakdown = GpccStreamBreakdown(
            sps_bytes=5,
            gps_bytes=5,
            slice_header_bytes=10,
            payload_bytes=payload_bytes,
            total_bytes=20 + payload_bytes,
        )
        stream = bytes([payload_bytes]) * breakdown.total_bytes
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "stream.bin").write_bytes(stream)
        return GpccResult(
            reconstruction=np.asarray(source, dtype=np.float32),
            stream_bytes=len(stream),
            stream_breakdown=breakdown,
            encode_seconds=0.01,
            decode_seconds=0.01,
            encoder_command=("fake",),
            decoder_command=("fake",),
            encoder_stdout="",
            decoder_stdout="",
        )

    monkeypatch.setattr(
        "pointconstellation.exp040_defect_codecs.run_tmc3", fake_run_tmc3
    )
    executable = tmp_path / "tmc3"
    executable.write_bytes(b"fake tmc3 executable")
    benchmark = DefectAnomalyBenchmarkConfig(
        num_points=64, output_dir=str(tmp_path / "output")
    )
    experiment_040 = SelectiveExperimentConfig(
        gpcc_reference_path=str(reference),
        tmc3_executable=str(executable),
        decoder_seeds=(7,),
    )
    frontier = _GpccFrontier(benchmark, experiment_040)
    points, _ = _sphere(64)

    stream_40 = frontier.encode(points, payload_budget=40, target_bytes=52)
    metadata_40 = frontier.metadata(stream_40, payload_budget=40)
    stream_60 = frontier.encode(points, payload_budget=60, target_bytes=80)

    assert metadata_40["codec_rate_point"] == "payload_35"
    assert metadata_40["codec_payload_bytes"] == 35
    assert metadata_40["payload_byte_delta"] == -5
    assert metadata_40["complete_stream_byte_delta"] == 3
    assert len(stream_60) == 90
    assert calls == ["payload_35", "payload_70"]
    decoded = frontier.decode(stream_40)
    assert sorted(map(tuple, decoded.tolist())) == sorted(map(tuple, points.tolist()))

    resumed_frontier = _GpccFrontier(benchmark, experiment_040)
    resumed_stream = resumed_frontier.encode(
        points, payload_budget=40, target_bytes=52
    )
    assert resumed_stream == stream_40
    assert calls == ["payload_35", "payload_70"]
    assert np.array_equal(resumed_frontier.decode(resumed_stream), decoded)


def test_smoke_resume_restores_byte_identical_rows_and_reports_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = DefectAnomalyBenchmarkConfig.from_json(
        Path("configs/experiment_041_defect_anomaly_smoke.json")
    )
    config = replace(
        config,
        output_dir=str(tmp_path / "resume_smoke"),
        codec_provider=None,
        diagnostic_subset_codecs=True,
        num_points=64,
        maximum_clouds_per_split=1,
        defect_types=("bump",),
        scorer_seeds=(4101,),
        scorer_points_per_reference=32,
        scorer_maximum_reference_clouds=2,
        scorer_cloud_candidates=2,
        distance_chunk_size=32,
        payload_budgets=(40,),
        primary_payload_budget=40,
        bootstrap_samples=100,
    )

    clean_result = run_defect_anomaly_benchmark(config, device_name="cpu")
    output = Path(config.output_dir)
    rows_path = output / "defect_per_cloud.jsonl"
    clean_rows = rows_path.read_bytes()
    lines = clean_rows.splitlines(keepends=True)
    assert len(lines) == len(clean_result["per_cloud"])
    rows_path.write_bytes(b"".join(lines[:-3]))
    capsys.readouterr()

    resumed_result = run_defect_anomaly_benchmark(config, device_name="cpu")
    progress_lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]

    assert rows_path.read_bytes() == clean_rows
    assert resumed_result["per_cloud"] == clean_result["per_cloud"]
    assert any(
        row["stage"] == "resume" and row["resume"] for row in progress_lines
    )
    scoring_progress = [
        row for row in progress_lines if row["stage"] == "scoring"
    ]
    assert scoring_progress
    assert scoring_progress[0]["resumed_scored_rows"] == len(lines) - 3
    assert scoring_progress[-1]["done"] == scoring_progress[-1]["total"]
    assert scoring_progress[-1]["persisted_path"] == str(rows_path)
    assert all(
        {"stage", "done", "total", "elapsed_seconds"} <= row.keys()
        for row in progress_lines
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["defect_seed"] == config.defect_seed
    assert manifest["config_sha256"] == resumed_result["config_sha256"]
    assert len(manifest["code_sha256"]) == 64
    assert manifest["model_hashes"] == {}
    assert manifest["status"] == "complete"
    assert set(resumed_result["timing_breakdown"]) == {
        "defect_injection_seconds",
        "scorer_fitting_seconds",
        "label_transfer_seconds",
        "scorer_inference_seconds",
        "point_metrics_seconds",
        "per_arm_encode_seconds",
        "per_arm_decode_seconds",
        "official_metrics_seconds",
        "bootstrap_seconds",
        "other_seconds",
        "total_seconds",
    }

    changed = replace(config, defect_seed=config.defect_seed + 1)
    changed_result = run_defect_anomaly_benchmark(changed, device_name="cpu")
    mismatch_progress = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert len(rows_path.read_bytes().splitlines()) == len(changed_result["per_cloud"])
    assert any(
        row["stage"] == "resume"
        and not row["resume"]
        and row["reason"] == "manifest_hash_mismatch"
        for row in mismatch_progress
    )


def test_codec_execution_never_receives_defect_labels() -> None:
    points, _ = _sphere(64)
    labels = np.zeros(len(points), dtype=np.uint8)
    labels[3:9] = 1

    class CoordinateOnlySpy:
        def __init__(self):
            self.encoded = []
            self.decoded = []

        def encode(self, source):
            self.encoded.append(np.asarray(source).copy())
            return b"source-only"

        def decode(self, stream):
            self.decoded.append(stream)
            return points.copy()

    codec = CoordinateOnlySpy()
    sample = BenchmarkCloud(
        split="validation",
        cloud_id="fixture:one",
        category="fixture",
        defect_type="dent",
        size_stratum="medium_2_4pct",
        declared_fraction=0.03,
        points=points,
        point_labels=labels,
        cloud_label=1,
        removed_count=0,
    )
    arm = CodecArm(
        name="constellation_only",
        payload_budget_bytes=1,
        target_bytes=len(b"source-only"),
        codec=codec,
    )

    decoded, checks = _decode_samples(
        [sample], [arm], expected_point_count=len(points)
    )

    assert len(codec.encoded) == len(codec.decoded) == 1
    assert np.array_equal(codec.encoded[0], points)
    assert codec.decoded == [b"source-only"]
    assert not np.array_equal(codec.encoded[0].reshape(-1)[: len(labels)], labels)
    assert decoded[1].sample.point_labels is labels
    assert all(checks.values())


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
