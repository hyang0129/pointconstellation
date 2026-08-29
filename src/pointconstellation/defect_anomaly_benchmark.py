"""Experiment 041: anomaly detection on decoded defected point clouds.

The primary runner is codec-agnostic.  Codec providers return objects exposing
only ``encode(source) -> bytes`` and ``decode(stream) -> points``; neither call
receives anomaly labels, normals, target samples, or scorer state.  The anomaly
scorer is fitted before any codec is called and sees only undefected raw clouds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.bitstream import (
    MODE_FIXED,
    decode_constellation,
    encode_constellation,
    expected_stream_bytes,
)
from pointconstellation.data import MeshSurfaceDataset, file_sha256
from pointconstellation.defects import (
    DEFECT_TYPES,
    DefectInjectionConfig,
    inject_defect_for_cloud,
    size_stratum,
    transfer_point_labels,
)

FloatArray = NDArray[np.float32]
EXPERIMENT_040_SELECTIVE_HEADER_BYTES = 16
EXPERIMENT_040_SCORE_METHODS = (
    "curvature",
    "density",
    "decoder_residual",
    "boundary",
)
RUN_MANIFEST_SCHEMA_VERSION = 2


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _source_tree_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class _ProgressReporter:
    def __init__(self, started: float) -> None:
        self.started = started

    def emit(self, stage: str, done: int, total: int, **details: Any) -> None:
        print(
            json.dumps(
                {
                    "stage": stage,
                    "done": done,
                    "total": total,
                    "elapsed_seconds": time.perf_counter() - self.started,
                    **details,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )


@runtime_checkable
class PointCloudCodec(Protocol):
    """The only Experiment 040 interface consumed by Experiment 041."""

    def encode(self, source: FloatArray) -> bytes:
        """Encode one source-visible coordinate array."""

    def decode(self, stream: bytes) -> FloatArray:
        """Decode one complete serialized stream to a coordinate array."""


@dataclass(frozen=True)
class CodecArm:
    """One named codec at one target byte point."""

    name: str
    payload_budget_bytes: int
    target_bytes: int
    codec: PointCloudCodec
    exact_bytes: bool = True
    maximum_rate_error_bytes: int = 0
    role: str = "coded"

    def __post_init__(self) -> None:
        if not self.name or self.name == "raw":
            raise ValueError("coded arm name must be nonempty and cannot be raw")
        if min(self.payload_budget_bytes, self.target_bytes) < 1:
            raise ValueError("payload budget and target bytes must be positive")
        if self.maximum_rate_error_bytes < 0:
            raise ValueError("maximum_rate_error_bytes cannot be negative")
        if self.exact_bytes and self.maximum_rate_error_bytes:
            raise ValueError("exact arms cannot declare a nonzero rate tolerance")
        if not isinstance(self.codec, PointCloudCodec):
            raise TypeError("codec must implement encode(source) and decode(stream)")


@dataclass(frozen=True)
class KNNScorerConfig:
    """Resource bounds for the non-learned normal-manifold scorer."""

    neighbors: int = 3
    points_per_reference: int = 512
    maximum_reference_clouds: int = 128
    cloud_candidates: int = 3
    tail_fraction: float = 0.02
    distance_chunk_size: int = 512

    def __post_init__(self) -> None:
        counts = (
            self.neighbors,
            self.points_per_reference,
            self.maximum_reference_clouds,
            self.cloud_candidates,
            self.distance_chunk_size,
        )
        if min(counts) < 1:
            raise ValueError("k-NN scorer counts must be positive")
        if self.cloud_candidates > self.maximum_reference_clouds:
            raise ValueError("cloud_candidates exceeds maximum_reference_clouds")
        if not 0.0 < self.tail_fraction <= 0.5:
            raise ValueError("tail_fraction must lie in (0, 0.5]")


@dataclass(frozen=True)
class DefectAnomalyBenchmarkConfig:
    """Validated data, rate, scorer, and gate protocol for Experiment 041."""

    dataset_root: str = "data/modelnet40_official/ModelNet40"
    dataset_manifest: str = "configs/manifests/modelnet40_stability.local.json"
    expected_manifest_sha256: str | None = None
    verify_mesh_hashes: bool = True
    num_points: int = 2048
    data_seed: int = 1517
    defect_seed: int = 20_260_841
    training_split: str = "train"
    evaluation_splits: tuple[str, ...] = ("validation", "category_ood")
    maximum_training_clouds: int | None = 128
    maximum_clouds_per_split: int | None = None
    defect_types: tuple[str, ...] = DEFECT_TYPES
    minimum_defect_fraction: float = 0.01
    maximum_defect_fraction: float = 0.05
    scorer_seeds: tuple[int, ...] = (4101, 4111, 4127)
    scorer_neighbors: int = 3
    scorer_points_per_reference: int = 512
    scorer_maximum_reference_clouds: int = 128
    scorer_cloud_candidates: int = 3
    scorer_tail_fraction: float = 0.02
    distance_chunk_size: int = 512
    payload_budgets: tuple[int, ...] = (40, 52, 64, 78, 96, 110)
    primary_payload_budget: int = 64
    experiment_040_config: str = "configs/experiment_040_selective.json"
    experiment_040_artifact_dir: str | None = "artifacts/local/experiment_040_selective"
    experiment_040_decoder_seed: int = 7
    selective_score_method: str = "decoder_residual"
    selective_preserved_fraction: float | None = 0.5
    codec_provider: str | None = None
    diagnostic_subset_codecs: bool = False
    gpcc_arm: str = "gpcc"
    gpcc_maximum_rate_error_bytes: int = 512
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_841
    confidence_level: float = 0.95
    primary_split: str = "validation"
    constellation_arm: str = "constellation_only"
    selective_arm: str = "selective_pass_through"
    random_control_arm: str = "selective_random_k2"
    external_manifest: str | None = None
    output_dir: str = "artifacts/local/experiment_041_defect_anomaly"

    def __post_init__(self) -> None:
        if self.num_points < 32:
            raise ValueError("num_points must be at least 32")
        if min(self.data_seed, self.defect_seed) < 0:
            raise ValueError("data and defect seeds must be nonnegative")
        if not self.training_split or not self.evaluation_splits:
            raise ValueError("training and evaluation splits must be nonempty")
        if len(set(self.evaluation_splits)) != len(self.evaluation_splits):
            raise ValueError("evaluation_splits must be unique")
        if self.training_split in self.evaluation_splits:
            raise ValueError("training split cannot also be an evaluation split")
        if self.primary_split not in self.evaluation_splits:
            raise ValueError("primary_split must be one of evaluation_splits")
        for count in (self.maximum_training_clouds, self.maximum_clouds_per_split):
            if count is not None and count < 1:
                raise ValueError("optional cloud limits must be positive")
        if (
            not self.defect_types
            or len(set(self.defect_types)) != len(self.defect_types)
            or not set(self.defect_types) <= set(DEFECT_TYPES)
        ):
            raise ValueError("defect_types must be a unique subset of DEFECT_TYPES")
        DefectInjectionConfig(
            minimum_fraction=self.minimum_defect_fraction,
            maximum_fraction=self.maximum_defect_fraction,
        )
        if not self.scorer_seeds or len(set(self.scorer_seeds)) != len(
            self.scorer_seeds
        ):
            raise ValueError("scorer_seeds must be nonempty and unique")
        if min(self.scorer_seeds) < 0:
            raise ValueError("scorer seeds must be nonnegative")
        KNNScorerConfig(
            neighbors=self.scorer_neighbors,
            points_per_reference=self.scorer_points_per_reference,
            maximum_reference_clouds=self.scorer_maximum_reference_clouds,
            cloud_candidates=self.scorer_cloud_candidates,
            tail_fraction=self.scorer_tail_fraction,
            distance_chunk_size=self.distance_chunk_size,
        )
        if (
            not self.payload_budgets
            or len(set(self.payload_budgets)) != len(self.payload_budgets)
            or tuple(sorted(self.payload_budgets)) != self.payload_budgets
            or min(self.payload_budgets) < 1
        ):
            raise ValueError("payload_budgets must be unique, increasing, and positive")
        if self.primary_payload_budget not in self.payload_budgets:
            raise ValueError("primary_payload_budget is absent from payload_budgets")
        if not self.experiment_040_config:
            raise ValueError("experiment_040_config must be nonempty")
        if self.experiment_040_decoder_seed < 0:
            raise ValueError("experiment_040_decoder_seed must be nonnegative")
        if self.selective_score_method not in EXPERIMENT_040_SCORE_METHODS:
            raise ValueError(
                "selective_score_method must be an Experiment 040 source score"
            )
        if self.selective_preserved_fraction is not None and not (
            0.0 < self.selective_preserved_fraction < 1.0
        ):
            raise ValueError("selective_preserved_fraction must lie in (0, 1)")
        arm_names = (
            self.constellation_arm,
            self.selective_arm,
            self.random_control_arm,
        )
        if any(not name or name == "raw" for name in arm_names):
            raise ValueError("gate arm names must be nonempty coded-arm names")
        if len(set(arm_names)) != len(arm_names):
            raise ValueError("gate arm names must be unique")
        if not self.gpcc_arm or self.gpcc_arm == "raw" or self.gpcc_arm in arm_names:
            raise ValueError("gpcc_arm must be a distinct coded-arm name")
        if self.gpcc_maximum_rate_error_bytes < 0:
            raise ValueError("G-PCC maximum rate error cannot be negative")
        if self.diagnostic_subset_codecs and self.codec_provider is not None:
            raise ValueError("choose a codec provider or diagnostic codecs, not both")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0.5, 1)")

    @property
    def scorer(self) -> KNNScorerConfig:
        return KNNScorerConfig(
            neighbors=self.scorer_neighbors,
            points_per_reference=self.scorer_points_per_reference,
            maximum_reference_clouds=self.scorer_maximum_reference_clouds,
            cloud_candidates=self.scorer_cloud_candidates,
            tail_fraction=self.scorer_tail_fraction,
            distance_chunk_size=self.distance_chunk_size,
        )

    @property
    def injection(self) -> DefectInjectionConfig:
        return DefectInjectionConfig(
            minimum_fraction=self.minimum_defect_fraction,
            maximum_fraction=self.maximum_defect_fraction,
        )

    @classmethod
    def from_json(cls, path: Path) -> DefectAnomalyBenchmarkConfig:
        values = json.loads(path.read_text())
        for key in (
            "evaluation_splits",
            "defect_types",
            "scorer_seeds",
            "payload_budgets",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def _run_identity(
    config: DefectAnomalyBenchmarkConfig,
    *,
    device_name: str,
    codec_override: bool,
) -> dict[str, Any]:
    config_values = asdict(config)
    dependency_hashes: dict[str, str] = {
        "dataset_manifest_sha256": file_sha256(Path(config.dataset_manifest)),
    }
    model_hashes: dict[str, str] = {}
    if (
        not codec_override
        and not config.diagnostic_subset_codecs
        and config.codec_provider
        == "pointconstellation.exp040_defect_codecs:build_codec_arms"
    ):
        experiment_040_path = Path(config.experiment_040_config)
        experiment_040 = json.loads(experiment_040_path.read_text())
        stability_path = Path(experiment_040["stability_config"])
        artifact_dir = Path(experiment_040["stability_artifact_dir"])
        decoder_dir = artifact_dir / (
            f"decoders/seed_{config.experiment_040_decoder_seed}"
        )
        dependency_hashes.update(
            {
                "experiment_040_config_sha256": file_sha256(experiment_040_path),
                "stability_config_sha256": file_sha256(stability_path),
                "gpcc_reference_sha256": file_sha256(
                    Path(experiment_040["gpcc_reference_path"])
                ),
                "tmc3_executable_sha256": file_sha256(
                    Path(experiment_040["tmc3_executable"])
                ),
            }
        )
        model_hashes.update(
            {
                "decoder_checkpoint_sha256": file_sha256(
                    decoder_dir / "stabilized.pt"
                ),
                "decoder_selection_sha256": file_sha256(
                    decoder_dir / "selection.json"
                ),
            }
        )
    identity = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "config_sha256": _json_sha256(config_values),
        "code_sha256": _source_tree_sha256(),
        "model_hashes": model_hashes,
        "dependency_hashes": dependency_hashes,
        "defect_seed": config.defect_seed,
        "device": device_name,
        "codec_override": codec_override,
    }
    identity["identity_sha256"] = _json_sha256(identity)
    return identity


def _prepare_run_manifest(
    config: DefectAnomalyBenchmarkConfig,
    *,
    device_name: str,
    codec_override: bool,
    output_dir: Path,
    progress: _ProgressReporter,
) -> tuple[dict[str, Any], bool]:
    identity = _run_identity(
        config, device_name=device_name, codec_override=codec_override
    )
    manifest_path = output_dir / "run_manifest.json"
    resume = False
    reason = "manifest_missing"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            reason = "manifest_unreadable"
        else:
            if existing.get("identity") == identity:
                resume = True
                reason = "identical_hashes"
            else:
                reason = "manifest_hash_mismatch"
    if not resume:
        for name in ("defect_per_cloud.jsonl", "defect_anomaly_metrics.json"):
            path = output_dir / name
            if path.exists():
                path.unlink()
    manifest = {
        "experiment": 41,
        "status": "running",
        "config": asdict(config),
        "config_sha256": identity["config_sha256"],
        "code_sha256": identity["code_sha256"],
        "model_hashes": identity["model_hashes"],
        "dependency_hashes": identity["dependency_hashes"],
        "defect_seed": config.defect_seed,
        "codec_provider": config.codec_provider,
        "diagnostic_subset_codecs": config.diagnostic_subset_codecs,
        "identity": identity,
    }
    _atomic_write_json(manifest_path, manifest)
    progress.emit(
        "resume",
        1 if resume else 0,
        1,
        resume=resume,
        reason=reason,
    )
    return manifest, resume


@dataclass(frozen=True)
class BenchmarkCloud:
    """One raw control or defect condition before codec evaluation."""

    split: str
    cloud_id: str
    category: str
    defect_type: str
    size_stratum: str
    declared_fraction: float
    points: FloatArray
    point_labels: NDArray[np.uint8]
    cloud_label: int
    removed_count: int
    domain_scale_factor: float = 1.0


@dataclass(frozen=True)
class DecodedCloud:
    """One codec output with nearest-sample proxy labels."""

    sample: BenchmarkCloud
    arm: str
    payload_budget_bytes: int | None
    target_bytes: int | None
    stream_bytes: int | None
    stream_sha256: str | None
    codec_header_bytes: int | None
    codec_payload_bytes: int | None
    payload_byte_delta: int | None
    complete_stream_byte_delta: int | None
    codec_rate_point: str | None
    decoded_points: FloatArray
    decoded_labels: NDArray[np.uint8]


@dataclass(frozen=True)
class ScoredCloud:
    """Metrics for one scorer seed, source condition, and codec arm."""

    scorer_seed: int
    split: str
    cloud_id: str
    category: str
    defect_type: str
    size_stratum: str
    declared_fraction: float
    domain_scale_factor: float
    cloud_label: int
    arm: str
    payload_budget_bytes: int | None
    target_bytes: int | None
    stream_bytes: int | None
    stream_sha256: str | None
    codec_header_bytes: int | None
    codec_payload_bytes: int | None
    payload_byte_delta: int | None
    complete_stream_byte_delta: int | None
    codec_rate_point: str | None
    decoded_point_count: int
    defective_point_count: int
    cloud_score: float
    point_auroc: float | None
    point_auprc: float | None


ScoredRowKey = tuple[int, str, str, str, str, int | None]
EvaluationUnitKey = tuple[str, str, str, str, int | None]


def _scored_row_key(row: ScoredCloud) -> ScoredRowKey:
    return (
        row.scorer_seed,
        row.split,
        row.cloud_id,
        row.defect_type,
        row.arm,
        row.payload_budget_bytes,
    )


def _evaluation_unit_key(row: ScoredCloud) -> EvaluationUnitKey:
    return (
        row.split,
        row.cloud_id,
        row.defect_type,
        row.arm,
        row.payload_budget_bytes,
    )


def _load_incremental_rows(path: Path) -> list[ScoredCloud]:
    if not path.is_file():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        boundary = payload.rfind(b"\n") + 1
        payload = payload[:boundary]
        path.write_bytes(payload)
    rows = []
    identities: set[ScoredRowKey] = set()
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            continue
        try:
            row = ScoredCloud(**json.loads(line))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid incremental row at {path}:{line_number}"
            ) from error
        identity = _scored_row_key(row)
        if identity in identities:
            raise ValueError(f"duplicate incremental row identity: {identity}")
        identities.add(identity)
        rows.append(row)
    return rows


def _append_incremental_row(handle: Any, row: ScoredCloud) -> None:
    handle.write(json.dumps(asdict(row)) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _point_array(points: ArrayLike) -> FloatArray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1:] != (3,) or not len(array):
        raise ValueError("point cloud must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all():
        raise ValueError("point cloud must contain finite coordinates")
    return array


def _cloud_descriptor(points: FloatArray) -> NDArray[np.float64]:
    values = points.astype(np.float64, copy=False)
    centered = values - values.mean(axis=0)
    radii = np.linalg.norm(centered, axis=1)
    covariance = centered.T @ centered / max(1, len(values))
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.sqrt(np.maximum(eigenvalues, 0.0).sum())), 1e-12)
    return np.concatenate(
        (
            values.mean(axis=0),
            values.std(axis=0),
            np.quantile(radii, (0.1, 0.25, 0.5, 0.75, 0.9)),
            eigenvalues / (scale * scale),
        )
    )


def _nearest_squared(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    chunk_size: int,
    neighbors: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if not 1 <= neighbors <= len(reference):
        raise ValueError("neighbors must be in [1, number of reference points]")
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
        if neighbors == 1:
            distances[start:stop] = squared[np.arange(len(rows)), selected]
        else:
            nearest = np.partition(squared, neighbors - 1, axis=1)[:, :neighbors]
            distances[start:stop] = nearest.mean(axis=1)
        indices[start:stop] = selected
    return distances, indices


class KNNNormalManifoldScorer:
    """Nearest-distance scorer fitted exclusively on undefected raw clouds.

    A compact global descriptor selects nearby normal training shapes.  Point
    scores then combine decoded-to-normal distance with normal-to-decoded
    distance attributed to the nearest decoded point.  The reverse term is what
    makes missing geometry such as a hole observable on a returned point set.
    """

    def __init__(self, config: KNNScorerConfig | None = None, *, seed: int = 0) -> None:
        if seed < 0:
            raise ValueError("scorer seed must be nonnegative")
        self.config = config or KNNScorerConfig()
        self.seed = seed
        self._references: tuple[FloatArray, ...] = ()
        self._descriptors: NDArray[np.float64] | None = None

    @property
    def fitted(self) -> bool:
        return bool(self._references)

    def fit(self, raw_normal_clouds: Sequence[ArrayLike]) -> KNNNormalManifoldScorer:
        """Fit from coordinate arrays only; labels and decodes are not accepted."""

        if not raw_normal_clouds:
            raise ValueError("normal-manifold fit requires at least one raw cloud")
        rng = np.random.default_rng(self.seed)
        cloud_indices = np.arange(len(raw_normal_clouds))
        if len(cloud_indices) > self.config.maximum_reference_clouds:
            cloud_indices = np.sort(
                rng.choice(
                    cloud_indices,
                    size=self.config.maximum_reference_clouds,
                    replace=False,
                )
            )
        references = []
        for cloud_index in cloud_indices:
            points = _point_array(raw_normal_clouds[int(cloud_index)])
            if len(points) > self.config.points_per_reference:
                selected = np.sort(
                    rng.choice(
                        len(points),
                        size=self.config.points_per_reference,
                        replace=False,
                    )
                )
                points = points[selected]
            references.append(points.copy())
        self._references = tuple(references)
        self._descriptors = np.stack([_cloud_descriptor(row) for row in references])
        return self

    def _candidate_indices(self, points: FloatArray) -> NDArray[np.int64]:
        if self._descriptors is None:
            raise RuntimeError("normal-manifold scorer must be fitted before scoring")
        descriptor = _cloud_descriptor(points)
        distances = np.sum((self._descriptors - descriptor[None]) ** 2, axis=1)
        count = min(self.config.cloud_candidates, len(distances))
        return np.argsort(distances, kind="stable")[:count].astype(np.int64)

    def score_points(self, points: ArrayLike) -> NDArray[np.float64]:
        """Return per-point distance to the selected finite normal manifolds."""

        query = _point_array(points)
        query64 = query.astype(np.float64, copy=False)
        candidates = self._candidate_indices(query)
        candidate_scores = []
        for candidate in candidates:
            reference = self._references[int(candidate)].astype(np.float64, copy=False)
            forward, _ = _nearest_squared(
                query64,
                reference,
                chunk_size=self.config.distance_chunk_size,
                neighbors=min(self.config.neighbors, len(reference)),
            )
            reverse, assigned = _nearest_squared(
                reference,
                query64,
                chunk_size=self.config.distance_chunk_size,
                neighbors=min(self.config.neighbors, len(query64)),
            )
            attributed = np.zeros(len(query), dtype=np.float64)
            np.maximum.at(attributed, assigned, reverse)
            candidate_scores.append(np.sqrt(np.maximum(forward, attributed)))
        scores = np.min(np.stack(candidate_scores), axis=0)
        if not np.isfinite(scores).all():
            raise RuntimeError("normal-manifold scorer produced a non-finite score")
        return scores

    def score_cloud(self, points: ArrayLike) -> float:
        scores = self.score_points(points)
        count = max(1, int(math.ceil(self.config.tail_fraction * len(scores))))
        return float(np.partition(scores, len(scores) - count)[-count:].mean())


def binary_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Compute tie-correct binary AUROC without a third-party dependency."""

    target = np.asarray(labels, dtype=np.uint8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(target) != len(values) or not len(target):
        raise ValueError("labels and scores must be nonempty and aligned")
    if np.any((target != 0) & (target != 1)) or not np.isfinite(values).all():
        raise ValueError("AUROC requires binary labels and finite scores")
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(ranks[target == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_auprc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Compute grouped-threshold binary average precision."""

    target = np.asarray(labels, dtype=np.uint8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(target) != len(values) or not len(target):
        raise ValueError("labels and scores must be nonempty and aligned")
    if np.any((target != 0) & (target != 1)) or not np.isfinite(values).all():
        raise ValueError("AUPRC requires binary labels and finite scores")
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-values, kind="stable")
    true_positives = 0
    predicted = 0
    weighted_precision = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        group_positives = int(target[order[start:stop]].sum())
        true_positives += group_positives
        predicted += stop - start
        weighted_precision += group_positives * true_positives / predicted
        start = stop
    return weighted_precision / positives


def assert_matched_bytes(
    streams: Mapping[str, bytes | bytearray | memoryview | int],
    *,
    target_bytes: int | None = None,
    tolerance_bytes: int = 0,
) -> int:
    """Assert actual serialized sizes are mutually matched and return the size."""

    if len(streams) < 2:
        raise ValueError("matched-byte assertion requires at least two coded arms")
    if tolerance_bytes < 0:
        raise ValueError("tolerance_bytes cannot be negative")
    sizes = {
        name: value if isinstance(value, int) else len(value)
        for name, value in streams.items()
    }
    if any(value < 1 for value in sizes.values()):
        raise ValueError("serialized streams must be nonempty")
    minimum = min(sizes.values())
    maximum = max(sizes.values())
    if maximum - minimum > tolerance_bytes:
        raise AssertionError(f"coded arms are not byte matched: {sizes}")
    if target_bytes is not None and any(
        abs(value - target_bytes) > tolerance_bytes for value in sizes.values()
    ):
        raise AssertionError(
            f"coded arms do not match target_bytes={target_bytes}: {sizes}"
        )
    return minimum


class _DiagnosticSubsetCodec:
    """Smoke-only exact-rate subset stream; never used for headline results."""

    def __init__(
        self,
        size: int,
        *,
        bits: int,
        mode: str,
        k2: int,
        complete_stream_bytes: int,
    ) -> None:
        self.size = size
        self.bits = bits
        self.mode = mode
        self.k2 = k2
        self.complete_stream_bytes = complete_stream_bytes

    @staticmethod
    def _fps(points: FloatArray, count: int) -> NDArray[np.int64]:
        centroid = points.mean(axis=0)
        distances = np.sum((points - centroid) ** 2, axis=1)
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], -distances))
        first = int(order[0])
        selected = [first]
        minimum = np.sum((points - points[first]) ** 2, axis=1)
        for _ in range(1, count):
            order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], -minimum))
            candidate = int(order[0])
            selected.append(candidate)
            minimum = np.minimum(
                minimum, np.sum((points - points[candidate]) ** 2, axis=1)
            )
        return np.asarray(selected, dtype=np.int64)

    def encode(self, source: FloatArray) -> bytes:
        points = _point_array(source)
        if self.size > len(points):
            raise ValueError("diagnostic subset size exceeds source cardinality")
        k2 = min(self.k2, self.size - 1)
        base = self._fps(points, self.size - k2)
        available = np.ones(len(points), dtype=bool)
        available[base] = False
        candidates = np.flatnonzero(available)
        if k2:
            if self.mode == "selective":
                center = points.mean(axis=0)
                scores = np.sum((points[candidates] - center) ** 2, axis=1)
                extra = candidates[
                    np.lexsort(
                        (
                            points[candidates, 2],
                            points[candidates, 1],
                            points[candidates, 0],
                            -scores,
                        )
                    )[:k2]
                ]
            elif self.mode == "random":
                canonical = candidates[
                    np.lexsort(
                        (
                            points[candidates, 2],
                            points[candidates, 1],
                            points[candidates, 0],
                        )
                    )
                ]
                point_order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
                digest = hashlib.sha256(points[point_order].tobytes()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
                extra = canonical[
                    np.sort(rng.choice(len(canonical), size=k2, replace=False))
                ]
            else:
                extra = self._fps(points, self.size)[-k2:]
            selected = np.concatenate((base, extra))
        else:
            selected = base
        coordinates = np.clip(points[selected], -1.0, 1.0)
        stream = encode_constellation(
            coordinates,
            bits=self.bits,
            mode=MODE_FIXED,
            output_points=self.size,
        )
        if len(stream) > self.complete_stream_bytes:
            raise RuntimeError("diagnostic stream exceeds its complete byte budget")
        return stream + bytes(self.complete_stream_bytes - len(stream))

    def decode(self, stream: bytes) -> FloatArray:
        if len(stream) != self.complete_stream_bytes:
            raise ValueError("diagnostic stream differs from its declared byte count")
        canonical_bytes = expected_stream_bytes(self.size, self.bits)
        if any(stream[canonical_bytes:]):
            raise ValueError("diagnostic rate-matching padding must contain zero bytes")
        return decode_constellation(stream[:canonical_bytes]).coordinates.astype(
            np.float32
        )


def build_diagnostic_subset_codec_arms(
    config: DefectAnomalyBenchmarkConfig,
) -> tuple[CodecArm, ...]:
    """Build exact-rate plumbing controls for the documented CPU smoke only."""

    arms = []
    for payload_budget in config.payload_budgets:
        size = (8 * payload_budget) // 36
        while size and math.ceil(36 * size / 8) > payload_budget:
            size -= 1
        if size < 2:
            raise ValueError("diagnostic payload budget cannot hold two points")
        # Experiment 040's selective fixed header is 16 bytes.  The diagnostic
        # uniform streams are padded inside this smoke-only adapter to exercise
        # equality of actual complete bytes, including all rate-matching bytes.
        target_bytes = EXPERIMENT_040_SELECTIVE_HEADER_BYTES + payload_budget
        k2 = max(1, size // 4)
        for name, mode in (
            (config.constellation_arm, "constellation"),
            (config.selective_arm, "selective"),
            (config.random_control_arm, "random"),
        ):
            arms.append(
                CodecArm(
                    name=name,
                    payload_budget_bytes=payload_budget,
                    target_bytes=target_bytes,
                    codec=_DiagnosticSubsetCodec(
                        size,
                        bits=12,
                        mode=mode,
                        k2=k2,
                        complete_stream_bytes=target_bytes,
                    ),
                    role="smoke_only_subset_diagnostic",
                )
            )
    return tuple(arms)


def _provider(
    config: DefectAnomalyBenchmarkConfig, device_name: str
) -> tuple[CodecArm, ...]:
    if config.diagnostic_subset_codecs:
        return build_diagnostic_subset_codec_arms(config)
    if config.codec_provider is None:
        raise ValueError(
            "full Experiment 041 requires codec_provider from Experiment 040"
        )
    if ":" not in config.codec_provider:
        raise ValueError("codec_provider must use module:function syntax")
    module_name, function_name = config.codec_provider.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    arms = function(config=config, device_name=device_name)
    return tuple(arms)


def _validate_codec_arms(
    arms: Sequence[CodecArm], config: DefectAnomalyBenchmarkConfig
) -> None:
    required_names = (
        config.constellation_arm,
        config.selective_arm,
        config.random_control_arm,
    )
    if not config.diagnostic_subset_codecs:
        required_names = (*required_names, config.gpcc_arm)
    expected = {
        (name, rate)
        for name in required_names
        for rate in config.payload_budgets
    }
    identities = [(arm.name, arm.payload_budget_bytes) for arm in arms]
    if len(set(identities)) != len(identities):
        raise ValueError("codec arm/rate identities must be unique")
    missing = expected - set(identities)
    if missing:
        raise ValueError(f"codec provider is missing required arms: {sorted(missing)}")
    unexpected_rates = {arm.payload_budget_bytes for arm in arms} - set(
        config.payload_budgets
    )
    if unexpected_rates:
        raise ValueError(
            f"codec provider returned undeclared rates: {unexpected_rates}"
        )


def _manifest_dataset(
    config: DefectAnomalyBenchmarkConfig, split: str
) -> MeshSurfaceDataset:
    return MeshSurfaceDataset(
        Path(config.dataset_root),
        Path(config.dataset_manifest),
        split=split,
        num_points=config.num_points,
        seed=config.data_seed,
        verify_hashes=config.verify_mesh_hashes,
        training_target="source",
    )


def _load_raw_clouds(
    config: DefectAnomalyBenchmarkConfig,
    *,
    progress: _ProgressReporter | None = None,
    timing: dict[str, Any] | None = None,
) -> tuple[list[FloatArray], list[BenchmarkCloud], dict[str, Any]]:
    manifest_path = Path(config.dataset_manifest)
    manifest_hash = file_sha256(manifest_path)
    if (
        config.expected_manifest_sha256 is not None
        and manifest_hash != config.expected_manifest_sha256
    ):
        raise ValueError("dataset manifest hash differs from expected_manifest_sha256")
    train_dataset = _manifest_dataset(config, config.training_split)
    training_count = len(train_dataset)
    if config.maximum_training_clouds is not None:
        training_count = min(training_count, config.maximum_training_clouds)
    normal_training = [
        train_dataset.sample(index).source_points.copy()
        for index in range(training_count)
    ]

    evaluation_datasets = []
    for split in config.evaluation_splits:
        dataset = _manifest_dataset(config, split)
        count = len(dataset)
        if config.maximum_clouds_per_split is not None:
            count = min(count, config.maximum_clouds_per_split)
        evaluation_datasets.append((split, dataset, count))

    samples = []
    memberships: dict[str, list[str]] = {config.training_split: []}
    for index in range(training_count):
        record = train_dataset.records[index]
        memberships[config.training_split].append(
            f"{record['category']}:{record['model_id']}"
        )
    injection_total = sum(
        count * (1 + len(config.defect_types))
        for _, _, count in evaluation_datasets
    )
    injection_done = 0
    if progress is not None:
        progress.emit("defect_injection", injection_done, injection_total)
    for split, dataset, count in evaluation_datasets:
        memberships[split] = []
        for index in range(count):
            raw = dataset.sample(index)
            cloud_id = f"{raw.category}:{raw.model_id}"
            memberships[split].append(cloud_id)
            injection_started = time.perf_counter()
            control = inject_defect_for_cloud(
                raw.source_points,
                "none",
                base_seed=config.defect_seed,
                cloud_id=cloud_id,
                normals=raw.source_normals,
                config=config.injection,
            )
            if timing is not None:
                timing["defect_injection_seconds"] += (
                    time.perf_counter() - injection_started
                )
            samples.append(
                BenchmarkCloud(
                    split=split,
                    cloud_id=cloud_id,
                    category=raw.category,
                    defect_type="none",
                    size_stratum="control",
                    declared_fraction=0.0,
                    domain_scale_factor=control.domain_scale_factor,
                    points=control.points,
                    point_labels=control.point_labels,
                    cloud_label=0,
                    removed_count=0,
                )
            )
            injection_done += 1
            if progress is not None:
                progress.emit(
                    "defect_injection",
                    injection_done,
                    injection_total,
                    split=split,
                    cloud_id=cloud_id,
                    defect_type="none",
                )
            for defect_type in config.defect_types:
                injection_started = time.perf_counter()
                defect = inject_defect_for_cloud(
                    raw.source_points,
                    defect_type,
                    base_seed=config.defect_seed,
                    cloud_id=cloud_id,
                    normals=raw.source_normals,
                    config=config.injection,
                )
                if timing is not None:
                    timing["defect_injection_seconds"] += (
                        time.perf_counter() - injection_started
                    )
                samples.append(
                    BenchmarkCloud(
                        split=split,
                        cloud_id=cloud_id,
                        category=raw.category,
                        defect_type=defect.defect_type,
                        size_stratum=size_stratum(defect.declared_fraction),
                        declared_fraction=defect.declared_fraction,
                        domain_scale_factor=defect.domain_scale_factor,
                        points=defect.points,
                        point_labels=defect.point_labels,
                        cloud_label=1,
                        removed_count=defect.removed_count,
                    )
                )
                injection_done += 1
                if progress is not None:
                    progress.emit(
                        "defect_injection",
                        injection_done,
                        injection_total,
                        split=split,
                        cloud_id=cloud_id,
                        defect_type=defect_type,
                    )
    flat = [value for rows in memberships.values() for value in rows]
    defect_samples = [sample for sample in samples if sample.cloud_label == 1]
    return (
        normal_training,
        samples,
        {
            "dataset": train_dataset.manifest["dataset"],
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "memberships": {
                split: {
                    "count": len(rows),
                    "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
                    "records": rows,
                }
                for split, rows in memberships.items()
            },
            "identities_unique": len(flat) == len(set(flat)),
            "defect_cardinality_policy": {
                "declared_source_points": config.num_points,
                "policy": "preserve_declared_regime_for_every_condition",
                "thin_spur_construction": "relocate_selected_source_patch",
                "hole_construction": (
                    "remove_patch_and_deterministically_resample_survivors"
                ),
                "all_conditions_match_declared_source_points": all(
                    len(sample.points) == config.num_points
                    and sample.point_labels.shape == (config.num_points,)
                    for sample in samples
                ),
            },
            "defect_domain_policy": {
                "declared_domain": [-1.0, 1.0],
                "policy": "uniform_displacement_attenuation_without_clipping",
                "attenuated_conditions": sum(
                    sample.domain_scale_factor < 1.0 for sample in defect_samples
                ),
                "minimum_domain_scale_factor": min(
                    (sample.domain_scale_factor for sample in defect_samples),
                    default=1.0,
                ),
            },
        },
    )


def _decode_samples(
    samples: Sequence[BenchmarkCloud],
    arms: Sequence[CodecArm],
    *,
    expected_point_count: int,
) -> tuple[list[DecodedCloud], dict[str, bool]]:
    decoded = []
    exact_matches = []
    within_tolerance = []
    accounting_checks = []
    codec_inputs_match_raw = []
    for sample in samples:
        if len(sample.points) != expected_point_count:
            raise ValueError(
                "benchmark source cardinality differs from the declared regime: "
                f"expected N={expected_point_count}, observed N={len(sample.points)} "
                f"for {sample.cloud_id}/{sample.defect_type}"
            )
        if sample.point_labels.shape != (expected_point_count,):
            raise ValueError(
                "benchmark source labels do not align with the declared regime"
            )
        decoded.append(
            DecodedCloud(
                sample=sample,
                arm="raw",
                payload_budget_bytes=None,
                target_bytes=None,
                stream_bytes=None,
                stream_sha256=None,
                codec_header_bytes=None,
                codec_payload_bytes=None,
                payload_byte_delta=None,
                complete_stream_byte_delta=None,
                codec_rate_point=None,
                decoded_points=sample.points.copy(),
                decoded_labels=sample.point_labels.copy(),
            )
        )
        exact_streams: dict[int, dict[str, bytes]] = {}
        for arm in arms:
            codec_source = sample.points.copy()
            codec_inputs_match_raw.append(np.array_equal(codec_source, sample.points))
            stream = arm.codec.encode(codec_source)
            if not isinstance(stream, bytes) or not stream:
                raise TypeError("codec encode must return nonempty bytes")
            actual_bytes = len(stream)
            error = abs(actual_bytes - arm.target_bytes)
            exact_matches.append(not arm.exact_bytes or error == 0)
            within_tolerance.append(
                arm.exact_bytes or error <= arm.maximum_rate_error_bytes
            )
            if arm.exact_bytes:
                exact_streams.setdefault(arm.target_bytes, {})[arm.name] = stream
            metadata: Mapping[str, Any] = {}
            metadata_function = getattr(arm.codec, "rate_metadata", None)
            if callable(metadata_function):
                raw_metadata = metadata_function(stream)
                if not isinstance(raw_metadata, Mapping):
                    raise TypeError("codec rate_metadata must return a mapping")
                metadata = raw_metadata
                header_bytes = metadata.get("codec_header_bytes")
                payload_bytes = metadata.get("codec_payload_bytes")
                payload_delta = metadata.get("payload_byte_delta")
                complete_delta = metadata.get("complete_stream_byte_delta")
                consistent = (
                    isinstance(header_bytes, int)
                    and isinstance(payload_bytes, int)
                    and header_bytes >= 0
                    and payload_bytes >= 0
                    and header_bytes + payload_bytes == actual_bytes
                    and payload_delta == payload_bytes - arm.payload_budget_bytes
                    and complete_delta == actual_bytes - arm.target_bytes
                )
                accounting_checks.append(consistent)
                if not consistent:
                    raise RuntimeError("codec returned inconsistent byte accounting")
            reconstructed = _point_array(arm.codec.decode(stream))
            labels = transfer_point_labels(
                sample.points,
                sample.point_labels,
                reconstructed,
            )
            decoded.append(
                DecodedCloud(
                    sample=sample,
                    arm=arm.name,
                    payload_budget_bytes=arm.payload_budget_bytes,
                    target_bytes=arm.target_bytes,
                    stream_bytes=actual_bytes,
                    stream_sha256=hashlib.sha256(stream).hexdigest(),
                    codec_header_bytes=metadata.get("codec_header_bytes"),
                    codec_payload_bytes=metadata.get("codec_payload_bytes"),
                    payload_byte_delta=metadata.get("payload_byte_delta"),
                    complete_stream_byte_delta=metadata.get(
                        "complete_stream_byte_delta"
                    ),
                    codec_rate_point=metadata.get("codec_rate_point"),
                    decoded_points=reconstructed.copy(),
                    decoded_labels=labels,
                )
            )
        for target_bytes, streams in exact_streams.items():
            if len(streams) >= 2:
                assert_matched_bytes(streams, target_bytes=target_bytes)
    return decoded, {
        "all_exact_arms_match_target_bytes": all(exact_matches),
        "all_nearest_rate_arms_within_declared_tolerance": all(within_tolerance),
        "all_declared_codec_byte_accounting_is_consistent": all(accounting_checks),
        "all_codec_inputs_equal_raw_input_coordinates": all(codec_inputs_match_raw),
    }


def _score_decodes(
    normal_training: Sequence[FloatArray],
    decoded: Sequence[DecodedCloud],
    config: DefectAnomalyBenchmarkConfig,
) -> list[ScoredCloud]:
    rows = []
    for scorer_seed in config.scorer_seeds:
        scorer = KNNNormalManifoldScorer(config.scorer, seed=scorer_seed).fit(
            normal_training
        )
        for row in decoded:
            scores = scorer.score_points(row.decoded_points)
            tail_count = max(
                1, int(math.ceil(config.scorer.tail_fraction * len(scores)))
            )
            cloud_score = float(
                np.partition(scores, len(scores) - tail_count)[-tail_count:].mean()
            )
            has_both_labels = (
                0 < int(row.decoded_labels.sum()) < len(row.decoded_labels)
            )
            rows.append(
                ScoredCloud(
                    scorer_seed=scorer_seed,
                    split=row.sample.split,
                    cloud_id=row.sample.cloud_id,
                    category=row.sample.category,
                    defect_type=row.sample.defect_type,
                    size_stratum=row.sample.size_stratum,
                    declared_fraction=row.sample.declared_fraction,
                    domain_scale_factor=row.sample.domain_scale_factor,
                    cloud_label=row.sample.cloud_label,
                    arm=row.arm,
                    payload_budget_bytes=row.payload_budget_bytes,
                    target_bytes=row.target_bytes,
                    stream_bytes=row.stream_bytes,
                    stream_sha256=row.stream_sha256,
                    codec_header_bytes=row.codec_header_bytes,
                    codec_payload_bytes=row.codec_payload_bytes,
                    payload_byte_delta=row.payload_byte_delta,
                    complete_stream_byte_delta=row.complete_stream_byte_delta,
                    codec_rate_point=row.codec_rate_point,
                    decoded_point_count=len(row.decoded_points),
                    defective_point_count=int(row.decoded_labels.sum()),
                    cloud_score=cloud_score,
                    point_auroc=(
                        binary_auroc(row.decoded_labels, scores)
                        if has_both_labels
                        else None
                    ),
                    point_auprc=(
                        binary_auprc(row.decoded_labels, scores)
                        if has_both_labels
                        else None
                    ),
                )
            )
    return rows


def _score_decode(
    row: DecodedCloud,
    *,
    scorer_seed: int,
    scorer: KNNNormalManifoldScorer,
    config: DefectAnomalyBenchmarkConfig,
) -> ScoredCloud:
    scores = scorer.score_points(row.decoded_points)
    tail_count = max(1, int(math.ceil(config.scorer.tail_fraction * len(scores))))
    cloud_score = float(
        np.partition(scores, len(scores) - tail_count)[-tail_count:].mean()
    )
    has_both_labels = 0 < int(row.decoded_labels.sum()) < len(row.decoded_labels)
    return ScoredCloud(
        scorer_seed=scorer_seed,
        split=row.sample.split,
        cloud_id=row.sample.cloud_id,
        category=row.sample.category,
        defect_type=row.sample.defect_type,
        size_stratum=row.sample.size_stratum,
        declared_fraction=row.sample.declared_fraction,
        domain_scale_factor=row.sample.domain_scale_factor,
        cloud_label=row.sample.cloud_label,
        arm=row.arm,
        payload_budget_bytes=row.payload_budget_bytes,
        target_bytes=row.target_bytes,
        stream_bytes=row.stream_bytes,
        stream_sha256=row.stream_sha256,
        codec_header_bytes=row.codec_header_bytes,
        codec_payload_bytes=row.codec_payload_bytes,
        payload_byte_delta=row.payload_byte_delta,
        complete_stream_byte_delta=row.complete_stream_byte_delta,
        codec_rate_point=row.codec_rate_point,
        decoded_point_count=len(row.decoded_points),
        defective_point_count=int(row.decoded_labels.sum()),
        cloud_score=cloud_score,
        point_auroc=(
            binary_auroc(row.decoded_labels, scores) if has_both_labels else None
        ),
        point_auprc=(
            binary_auprc(row.decoded_labels, scores) if has_both_labels else None
        ),
    )


def _raw_decode(sample: BenchmarkCloud) -> DecodedCloud:
    return DecodedCloud(
        sample=sample,
        arm="raw",
        payload_budget_bytes=None,
        target_bytes=None,
        stream_bytes=None,
        stream_sha256=None,
        codec_header_bytes=None,
        codec_payload_bytes=None,
        payload_byte_delta=None,
        complete_stream_byte_delta=None,
        codec_rate_point=None,
        decoded_points=sample.points.copy(),
        decoded_labels=sample.point_labels.copy(),
    )


def _coded_decode(
    sample: BenchmarkCloud,
    arm: CodecArm,
    *,
    expected_point_count: int,
    timing: dict[str, Any],
) -> DecodedCloud:
    if len(sample.points) != expected_point_count:
        raise ValueError(
            "benchmark source cardinality differs from the declared regime: "
            f"expected N={expected_point_count}, observed N={len(sample.points)} "
            f"for {sample.cloud_id}/{sample.defect_type}"
        )
    if sample.point_labels.shape != (expected_point_count,):
        raise ValueError("benchmark source labels do not align with declared regime")
    codec_source = sample.points.copy()
    if not np.array_equal(codec_source, sample.points):
        raise RuntimeError("codec input differs from raw benchmark coordinates")
    encode_started = time.perf_counter()
    stream = arm.codec.encode(codec_source)
    timing["per_arm_encode_seconds"][arm.name] += (
        time.perf_counter() - encode_started
    )
    if not isinstance(stream, bytes) or not stream:
        raise TypeError("codec encode must return nonempty bytes")
    actual_bytes = len(stream)
    error = abs(actual_bytes - arm.target_bytes)
    if arm.exact_bytes and error:
        raise AssertionError(
            f"exact codec arm {arm.name} differs from target bytes: "
            f"{actual_bytes} != {arm.target_bytes}"
        )
    if not arm.exact_bytes and error > arm.maximum_rate_error_bytes:
        raise AssertionError(
            f"codec arm {arm.name} exceeds its declared rate tolerance"
        )
    metadata: Mapping[str, Any] = {}
    metadata_function = getattr(arm.codec, "rate_metadata", None)
    if callable(metadata_function):
        raw_metadata = metadata_function(stream)
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("codec rate_metadata must return a mapping")
        metadata = raw_metadata
        header_bytes = metadata.get("codec_header_bytes")
        payload_bytes = metadata.get("codec_payload_bytes")
        consistent = (
            isinstance(header_bytes, int)
            and isinstance(payload_bytes, int)
            and header_bytes >= 0
            and payload_bytes >= 0
            and header_bytes + payload_bytes == actual_bytes
            and metadata.get("payload_byte_delta")
            == payload_bytes - arm.payload_budget_bytes
            and metadata.get("complete_stream_byte_delta")
            == actual_bytes - arm.target_bytes
        )
        if not consistent:
            raise RuntimeError("codec returned inconsistent byte accounting")
    decode_started = time.perf_counter()
    reconstructed = _point_array(arm.codec.decode(stream))
    labels = transfer_point_labels(
        sample.points,
        sample.point_labels,
        reconstructed,
    )
    timing["per_arm_decode_seconds"][arm.name] += (
        time.perf_counter() - decode_started
    )
    return DecodedCloud(
        sample=sample,
        arm=arm.name,
        payload_budget_bytes=arm.payload_budget_bytes,
        target_bytes=arm.target_bytes,
        stream_bytes=actual_bytes,
        stream_sha256=hashlib.sha256(stream).hexdigest(),
        codec_header_bytes=metadata.get("codec_header_bytes"),
        codec_payload_bytes=metadata.get("codec_payload_bytes"),
        payload_byte_delta=metadata.get("payload_byte_delta"),
        complete_stream_byte_delta=metadata.get("complete_stream_byte_delta"),
        codec_rate_point=metadata.get("codec_rate_point"),
        decoded_points=reconstructed.copy(),
        decoded_labels=labels,
    )


def _evaluation_units(
    samples: Sequence[BenchmarkCloud], arms: Sequence[CodecArm]
) -> list[tuple[BenchmarkCloud, CodecArm | None]]:
    return [
        (sample, arm)
        for sample in samples
        for arm in (None, *arms)
    ]


def _unit_identity(
    sample: BenchmarkCloud, arm: CodecArm | None
) -> EvaluationUnitKey:
    return (
        sample.split,
        sample.cloud_id,
        sample.defect_type,
        "raw" if arm is None else arm.name,
        None if arm is None else arm.payload_budget_bytes,
    )


def _validate_incremental_rows(
    rows: Sequence[ScoredCloud],
    *,
    units: Sequence[tuple[BenchmarkCloud, CodecArm | None]],
    config: DefectAnomalyBenchmarkConfig,
) -> None:
    sample_by_identity = {
        (sample.split, sample.cloud_id, sample.defect_type): sample
        for sample, _ in units
    }
    expected = [
        (seed, *_unit_identity(sample, arm))
        for sample, arm in units
        for seed in config.scorer_seeds
    ]
    observed = [_scored_row_key(row) for row in rows]
    if observed != expected[: len(observed)]:
        raise ValueError("incremental rows are not a canonical completed prefix")
    for row in rows:
        sample = sample_by_identity[(row.split, row.cloud_id, row.defect_type)]
        if (
            row.category != sample.category
            or row.size_stratum != sample.size_stratum
            or row.declared_fraction != sample.declared_fraction
            or row.domain_scale_factor != sample.domain_scale_factor
            or row.cloud_label != sample.cloud_label
        ):
            raise ValueError("incremental row metadata differs from benchmark source")


def _rate_checks_from_rows(
    rows: Sequence[ScoredCloud],
    *,
    samples: Sequence[BenchmarkCloud],
    arms: Sequence[CodecArm],
    scorer_seed: int,
) -> dict[str, bool]:
    representatives = [row for row in rows if row.scorer_seed == scorer_seed]
    arm_by_identity = {
        (arm.name, arm.payload_budget_bytes): arm for arm in arms
    }
    exact_matches = []
    within_tolerance = []
    accounting_checks = []
    for row in representatives:
        if row.arm == "raw":
            continue
        arm = arm_by_identity[(row.arm, row.payload_budget_bytes)]
        error = abs(int(row.stream_bytes) - arm.target_bytes)
        exact_matches.append(not arm.exact_bytes or error == 0)
        within_tolerance.append(
            arm.exact_bytes or error <= arm.maximum_rate_error_bytes
        )
        accounting = (
            row.codec_header_bytes,
            row.codec_payload_bytes,
            row.payload_byte_delta,
            row.complete_stream_byte_delta,
        )
        if all(value is None for value in accounting):
            continue
        accounting_checks.append(
            isinstance(row.codec_header_bytes, int)
            and isinstance(row.codec_payload_bytes, int)
            and row.codec_header_bytes >= 0
            and row.codec_payload_bytes >= 0
            and row.codec_header_bytes + row.codec_payload_bytes == row.stream_bytes
            and row.payload_byte_delta
            == row.codec_payload_bytes - int(row.payload_budget_bytes)
            and row.complete_stream_byte_delta == row.stream_bytes - row.target_bytes
        )
    for sample in samples:
        sample_rows = [
            row
            for row in representatives
            if row.split == sample.split
            and row.cloud_id == sample.cloud_id
            and row.defect_type == sample.defect_type
        ]
        exact_streams: dict[int, dict[str, int]] = {}
        for row in sample_rows:
            if row.arm == "raw":
                continue
            arm = arm_by_identity[(row.arm, row.payload_budget_bytes)]
            if arm.exact_bytes:
                exact_streams.setdefault(arm.target_bytes, {})[row.arm] = int(
                    row.stream_bytes
                )
        for target_bytes, streams in exact_streams.items():
            if len(streams) >= 2:
                assert_matched_bytes(streams, target_bytes=target_bytes)
    return {
        "all_exact_arms_match_target_bytes": all(exact_matches),
        "all_nearest_rate_arms_within_declared_tolerance": all(within_tolerance),
        "all_declared_codec_byte_accounting_is_consistent": all(accounting_checks),
        "all_codec_inputs_equal_raw_input_coordinates": True,
    }


def _evaluate_incrementally(
    normal_training: Sequence[FloatArray],
    samples: Sequence[BenchmarkCloud],
    arms: Sequence[CodecArm],
    config: DefectAnomalyBenchmarkConfig,
    *,
    output_dir: Path,
    resume: bool,
    progress: _ProgressReporter,
    timing: dict[str, Any],
) -> tuple[list[ScoredCloud], dict[str, bool]]:
    path = output_dir / "defect_per_cloud.jsonl"
    loaded = _load_incremental_rows(path) if resume else []
    units = _evaluation_units(samples, arms)
    try:
        _validate_incremental_rows(loaded, units=units, config=config)
    except ValueError as error:
        loaded = []
        path.write_bytes(b"")
        progress.emit(
            "resume_rows",
            0,
            len(units),
            resume=False,
            reason="invalid_incremental_rows_clean_start",
            detail=str(error),
        )
    rows_by_key = {_scored_row_key(row): row for row in loaded}
    expected_keys = [
        (seed, *_unit_identity(sample, arm))
        for sample, arm in units
        for seed in config.scorer_seeds
    ]
    completed_units = len(loaded) // len(config.scorer_seeds)
    progress.emit(
        "evaluation",
        completed_units,
        len(units),
        resumed_scored_rows=len(loaded),
    )

    pending = len(loaded) < len(expected_keys)
    scorers: dict[int, KNNNormalManifoldScorer] = {}
    progress.emit("scorer_fitting", 0, len(config.scorer_seeds), skipped=not pending)
    if pending:
        for index, scorer_seed in enumerate(config.scorer_seeds, start=1):
            fit_started = time.perf_counter()
            scorers[scorer_seed] = KNNNormalManifoldScorer(
                config.scorer, seed=scorer_seed
            ).fit(normal_training)
            timing["scorer_fitting_seconds"] += time.perf_counter() - fit_started
            progress.emit(
                "scorer_fitting",
                index,
                len(config.scorer_seeds),
                scorer_seed=scorer_seed,
            )
    else:
        progress.emit(
            "scorer_fitting",
            len(config.scorer_seeds),
            len(config.scorer_seeds),
            skipped=True,
        )

    with path.open("a") as handle:
        for unit_index, (sample, arm) in enumerate(units, start=1):
            unit = _unit_identity(sample, arm)
            unit_keys = [(seed, *unit) for seed in config.scorer_seeds]
            if all(key in rows_by_key for key in unit_keys):
                continue
            progress.emit(
                "evaluation",
                unit_index - 1,
                len(units),
                status="started",
                split=sample.split,
                cloud_id=sample.cloud_id,
                defect_type=sample.defect_type,
                arm="raw" if arm is None else arm.name,
                payload_budget_bytes=(
                    None if arm is None else arm.payload_budget_bytes
                ),
            )
            decode_started = time.perf_counter()
            if arm is None:
                decoded = _raw_decode(sample)
                timing["per_arm_decode_seconds"]["raw"] += (
                    time.perf_counter() - decode_started
                )
            else:
                decoded = _coded_decode(
                    sample,
                    arm,
                    expected_point_count=config.num_points,
                    timing=timing,
                )
            for scorer_seed, key in zip(
                config.scorer_seeds, unit_keys, strict=True
            ):
                if key in rows_by_key:
                    continue
                metric_started = time.perf_counter()
                row = _score_decode(
                    decoded,
                    scorer_seed=scorer_seed,
                    scorer=scorers[scorer_seed],
                    config=config,
                )
                timing["official_metrics_seconds"] += (
                    time.perf_counter() - metric_started
                )
                _append_incremental_row(handle, row)
                rows_by_key[key] = row
            completed_units = unit_index
            progress.emit(
                "evaluation",
                completed_units,
                len(units),
                split=sample.split,
                cloud_id=sample.cloud_id,
                defect_type=sample.defect_type,
                arm="raw" if arm is None else arm.name,
                payload_budget_bytes=(
                    None if arm is None else arm.payload_budget_bytes
                ),
            )
    if set(rows_by_key) != set(expected_keys):
        raise RuntimeError("incremental evaluation did not produce every expected row")
    scored = [rows_by_key[key] for key in expected_keys]
    return scored, _rate_checks_from_rows(
        scored,
        samples=samples,
        arms=arms,
        scorer_seed=config.scorer_seeds[0],
    )


def _filtered_rows(
    rows: Sequence[ScoredCloud],
    *,
    arm: str,
    rate: int | None,
    split: str,
    defect_type: str | None = None,
    stratum: str | None = None,
) -> list[ScoredCloud]:
    selected = [
        row
        for row in rows
        if row.arm == arm and row.payload_budget_bytes == rate and row.split == split
    ]
    if defect_type is not None:
        selected = [row for row in selected if row.defect_type in {"none", defect_type}]
    if stratum is not None:
        positive_clouds = {
            (row.scorer_seed, row.cloud_id)
            for row in selected
            if row.cloud_label == 1 and row.size_stratum == stratum
        }
        selected = [
            row
            for row in selected
            if (row.cloud_label == 1 and row.size_stratum == stratum)
            or (
                row.cloud_label == 0
                and (row.scorer_seed, row.cloud_id) in positive_clouds
            )
        ]
    return selected


def _metric(rows: Sequence[ScoredCloud], metric: str) -> float:
    if metric == "cloud_auroc":
        return binary_auroc(
            [row.cloud_label for row in rows], [row.cloud_score for row in rows]
        )
    if metric == "cloud_auprc":
        return binary_auprc(
            [row.cloud_label for row in rows], [row.cloud_score for row in rows]
        )
    if metric not in {"point_auroc", "point_auprc"}:
        raise ValueError(f"unknown anomaly metric: {metric}")
    values = [getattr(row, metric) for row in rows if getattr(row, metric) is not None]
    return float(np.mean(values)) if values else float("nan")


def _point_estimate(rows: Sequence[ScoredCloud], metric: str) -> float:
    per_seed = []
    for seed in sorted({row.scorer_seed for row in rows}):
        value = _metric([row for row in rows if row.scorer_seed == seed], metric)
        if np.isfinite(value):
            per_seed.append(value)
    return float(np.mean(per_seed)) if per_seed else float("nan")


def _bootstrap_metric(
    rows: Sequence[ScoredCloud],
    *,
    metric: str,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    seeds = sorted({row.scorer_seed for row in rows})
    clouds = sorted({row.cloud_id for row in rows})
    if not seeds or not clouds:
        raise ValueError("bootstrap requires scorer seeds and clouds")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        seed_draw = rng.choice(seeds, size=len(seeds), replace=True)
        cloud_draw = rng.choice(clouds, size=len(clouds), replace=True)
        seed_values = []
        for scorer_seed in seed_draw:
            sampled_rows = []
            for cloud_id in cloud_draw:
                sampled_rows.extend(
                    row
                    for row in rows
                    if row.scorer_seed == scorer_seed and row.cloud_id == cloud_id
                )
            seed_values.append(_metric(sampled_rows, metric))
        finite_seed_values = [value for value in seed_values if np.isfinite(value)]
        draws[index] = (
            float(np.mean(finite_seed_values)) if finite_seed_values else float("nan")
        )
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return {
            "mean": None,
            "confidence_interval_lower": None,
            "confidence_interval_upper": None,
            "confidence_level": confidence_level,
            "bootstrap_samples": samples,
        }
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(finite, (alpha, 1.0 - alpha))
    estimate = _point_estimate(rows, metric)
    return {
        "mean": float(estimate),
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
    }


def _paired_bootstrap_difference(
    candidate: Sequence[ScoredCloud],
    baseline: Sequence[ScoredCloud],
    *,
    metric: str,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    candidate_identity = {
        (row.scorer_seed, row.cloud_id, row.defect_type) for row in candidate
    }
    baseline_identity = {
        (row.scorer_seed, row.cloud_id, row.defect_type) for row in baseline
    }
    if candidate_identity != baseline_identity:
        raise ValueError("paired arm rows do not share scorer/cloud/condition identity")
    seeds = sorted({row.scorer_seed for row in candidate})
    clouds = sorted({row.cloud_id for row in candidate})
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        seed_draw = rng.choice(seeds, size=len(seeds), replace=True)
        cloud_draw = rng.choice(clouds, size=len(clouds), replace=True)
        seed_differences = []
        for scorer_seed in seed_draw:
            candidate_draw = []
            baseline_draw = []
            for cloud_id in cloud_draw:
                candidate_draw.extend(
                    row
                    for row in candidate
                    if row.scorer_seed == scorer_seed and row.cloud_id == cloud_id
                )
                baseline_draw.extend(
                    row
                    for row in baseline
                    if row.scorer_seed == scorer_seed and row.cloud_id == cloud_id
                )
            seed_differences.append(
                _metric(candidate_draw, metric) - _metric(baseline_draw, metric)
            )
        finite_differences = [value for value in seed_differences if np.isfinite(value)]
        draws[index] = (
            float(np.mean(finite_differences)) if finite_differences else float("nan")
        )
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return {
            "candidate": candidate[0].arm,
            "baseline": baseline[0].arm,
            "metric": metric,
            "difference": None,
            "confidence_interval_lower": None,
            "confidence_interval_upper": None,
            "confidence_level": confidence_level,
            "bootstrap_samples": samples,
            "positive": False,
            "status": "unavailable_no_finite_point_metrics",
        }
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(finite, (alpha, 1.0 - alpha))
    difference = _point_estimate(candidate, metric) - _point_estimate(baseline, metric)
    return {
        "candidate": candidate[0].arm,
        "baseline": baseline[0].arm,
        "metric": metric,
        "difference": float(difference),
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "positive": bool(lower > 0.0),
    }


def summarize_anomaly_rows(
    rows: Sequence[ScoredCloud], config: DefectAnomalyBenchmarkConfig
) -> dict[str, Any]:
    """Aggregate headline, defect-stratified, and size-stratified metrics."""

    arms_and_rates = sorted(
        {(row.arm, row.payload_budget_bytes) for row in rows},
        key=lambda item: (item[1] is not None, item[1] or 0, item[0]),
    )
    summaries = []
    summary_index = 0
    for split in config.evaluation_splits:
        for arm, rate in arms_and_rates:
            base = _filtered_rows(rows, arm=arm, rate=rate, split=split)
            if not base:
                continue
            for stratifier, value in [
                (None, None),
                *[("defect_type", name) for name in config.defect_types],
                *[
                    ("size_stratum", name)
                    for name in ("small_1_2pct", "medium_2_4pct", "large_4_5pct")
                ],
            ]:
                selected = base
                if stratifier == "defect_type":
                    selected = _filtered_rows(
                        rows,
                        arm=arm,
                        rate=rate,
                        split=split,
                        defect_type=value,
                    )
                elif stratifier == "size_stratum":
                    selected = _filtered_rows(
                        rows,
                        arm=arm,
                        rate=rate,
                        split=split,
                        stratum=value,
                    )
                if not selected or not any(row.cloud_label for row in selected):
                    continue
                metrics = {}
                for metric_index, metric in enumerate(
                    ("cloud_auroc", "cloud_auprc", "point_auroc", "point_auprc")
                ):
                    metrics[metric] = _bootstrap_metric(
                        selected,
                        metric=metric,
                        samples=config.bootstrap_samples,
                        confidence_level=config.confidence_level,
                        seed=(
                            config.bootstrap_seed
                            + 10_000 * summary_index
                            + 100 * metric_index
                        ),
                    )
                summaries.append(
                    {
                        "split": split,
                        "arm": arm,
                        "payload_budget_bytes": rate,
                        "target_bytes": (
                            selected[0].target_bytes if rate is not None else None
                        ),
                        "stratifier": stratifier,
                        "stratum": value,
                        "clouds": len({row.cloud_id for row in selected}),
                        "scorer_seeds": sorted({row.scorer_seed for row in selected}),
                        "metrics": metrics,
                    }
                )
                summary_index += 1
    return {"rows": summaries}


def compute_gate_g_c2(
    rows: Sequence[ScoredCloud], config: DefectAnomalyBenchmarkConfig
) -> dict[str, Any]:
    """Compute the predeclared primary selective-recovery gate."""

    comparisons = []
    candidate = _filtered_rows(
        rows,
        arm=config.selective_arm,
        rate=config.primary_payload_budget,
        split=config.primary_split,
    )
    for baseline_index, baseline_arm in enumerate(
        (config.constellation_arm, config.random_control_arm)
    ):
        baseline = _filtered_rows(
            rows,
            arm=baseline_arm,
            rate=config.primary_payload_budget,
            split=config.primary_split,
        )
        for metric_index, metric in enumerate(("cloud_auroc", "point_auroc")):
            comparisons.append(
                _paired_bootstrap_difference(
                    candidate,
                    baseline,
                    metric=metric,
                    samples=config.bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=(
                        config.bootstrap_seed
                        + 1_000_000
                        + 10_000 * baseline_index
                        + metric_index
                    ),
                )
            )
    passed = all(row["positive"] for row in comparisons)
    return {
        "name": "G-C2",
        "passed": passed,
        "primary_split": config.primary_split,
        "primary_payload_budget": config.primary_payload_budget,
        "candidate": config.selective_arm,
        "required_baselines": [config.constellation_arm, config.random_control_arm],
        "required_metrics": ["cloud_auroc", "point_auroc"],
        "rule": (
            "all paired hierarchical scorer-seed/cloud bootstrap lower bounds "
            "for candidate-minus-baseline AUROC must exceed zero"
        ),
        "comparisons": comparisons,
    }


def load_external_anomaly_manifest(path: Path) -> dict[str, Any]:
    """Validate, but do not download, an MVTec 3D-AD/Real3D-AD manifest hook."""

    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("external anomaly manifest version must be 1")
    if manifest.get("dataset") not in {"MVTec 3D-AD", "Real3D-AD"}:
        raise ValueError("external anomaly dataset must be MVTec 3D-AD or Real3D-AD")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("external anomaly manifest must contain nonempty splits")
    for split, records in splits.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f"external anomaly split must be nonempty: {split}")
        for record in records:
            required = {
                "model_id",
                "category",
                "pointcloud",
                "pointcloud_sha256",
                "cloud_label",
            }
            if not isinstance(record, dict) or not required <= record.keys():
                raise ValueError(f"invalid external anomaly record in split {split}")
            if record["cloud_label"] not in {0, 1}:
                raise ValueError("external anomaly cloud_label must be binary")
    return manifest


def run_defect_anomaly_benchmark(
    config: DefectAnomalyBenchmarkConfig,
    *,
    device_name: str,
    codec_arms: Sequence[CodecArm] | None = None,
) -> dict[str, Any]:
    """Run deterministic injection, codec evaluation, scoring, and G-C2."""

    if device_name not in {"cpu", "mps", "cuda"}:
        raise ValueError("device_name must be cpu, mps, or cuda")
    started = time.perf_counter()
    progress = _ProgressReporter(started)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest, resume = _prepare_run_manifest(
        config,
        device_name=device_name,
        codec_override=codec_arms is not None,
        output_dir=output_dir,
        progress=progress,
    )
    if config.external_manifest is not None:
        load_external_anomaly_manifest(Path(config.external_manifest))

    timing: dict[str, Any] = {
        "defect_injection_seconds": 0.0,
        "scorer_fitting_seconds": 0.0,
        "per_arm_encode_seconds": {},
        "per_arm_decode_seconds": {"raw": 0.0},
        "official_metrics_seconds": 0.0,
        "bootstrap_seconds": 0.0,
    }
    normal_training, samples, data_protocol = _load_raw_clouds(
        config, progress=progress, timing=timing
    )
    arms = (
        tuple(codec_arms) if codec_arms is not None else _provider(config, device_name)
    )
    _validate_codec_arms(arms, config)
    timing["per_arm_encode_seconds"] = {
        name: 0.0 for name in sorted({arm.name for arm in arms})
    }
    timing["per_arm_decode_seconds"].update(
        {name: 0.0 for name in sorted({arm.name for arm in arms})}
    )
    scored, rate_checks = _evaluate_incrementally(
        normal_training,
        samples,
        arms,
        config,
        output_dir=output_dir,
        resume=resume,
        progress=progress,
        timing=timing,
    )
    progress.emit("bootstrap", 0, 2)
    bootstrap_started = time.perf_counter()
    summaries = summarize_anomaly_rows(scored, config)
    progress.emit("bootstrap", 1, 2, component="summaries")
    gate = compute_gate_g_c2(scored, config)
    timing["bootstrap_seconds"] += time.perf_counter() - bootstrap_started
    progress.emit("bootstrap", 2, 2, component="gate_g_c2")
    contract_checks = {
        "scorer_fit_uses_raw_normal_training_clouds_only": True,
        "codec_encode_receives_coordinates_only": True,
        "codec_decode_receives_stream_only": True,
        "every_cloud_has_undefected_control": all(
            any(
                row.cloud_id == cloud_id
                and row.split == split
                and row.defect_type == "none"
                for row in samples
            )
            for split, cloud_id in {(row.split, row.cloud_id) for row in samples}
        ),
        "defect_fractions_within_declared_bounds": all(
            row.cloud_label == 0
            or config.minimum_defect_fraction
            <= row.declared_fraction
            <= config.maximum_defect_fraction
            for row in samples
        ),
        "all_injected_clouds_respect_declared_codec_domain": all(
            np.all(row.points >= -1.0) and np.all(row.points <= 1.0)
            for row in samples
        ),
        "all_benchmark_sources_match_declared_cardinality": all(
            len(row.points) == config.num_points
            and row.point_labels.shape == (config.num_points,)
            for row in samples
        ),
        "nearest_target_point_label_transfer_is_explicit": True,
        **rate_checks,
        "data_memberships_are_disjoint": data_protocol["identities_unique"],
    }
    if not all(contract_checks.values()):
        raise RuntimeError("Experiment 041 scientific contract failed")

    per_cloud_rows = [asdict(row) for row in scored]
    elapsed_seconds = time.perf_counter() - started
    accounted_seconds = (
        timing["defect_injection_seconds"]
        + timing["scorer_fitting_seconds"]
        + sum(timing["per_arm_encode_seconds"].values())
        + sum(timing["per_arm_decode_seconds"].values())
        + timing["official_metrics_seconds"]
        + timing["bootstrap_seconds"]
    )
    timing["other_seconds"] = max(0.0, elapsed_seconds - accounted_seconds)
    timing["total_seconds"] = elapsed_seconds
    result = {
        "experiment": 41,
        "status": "smoke_only" if config.diagnostic_subset_codecs else "complete",
        "config": asdict(config),
        "config_sha256": run_manifest["config_sha256"],
        "device": device_name,
        "data_protocol": data_protocol,
        "scorer": {
            "kind": "nonlearned_knn_distance_to_normal_manifold",
            "training_input": "undefected_raw_coordinates_only",
            "seeds": list(config.scorer_seeds),
        },
        "codec_protocol": "encode(source) -> bytes; decode(bytes) -> points",
        "experiment_040_selection": {
            "artifact_dir": config.experiment_040_artifact_dir,
            "score_method": config.selective_score_method,
            "preserved_fraction": config.selective_preserved_fraction,
        },
        "codec_arms": [
            {
                "name": arm.name,
                "payload_budget_bytes": arm.payload_budget_bytes,
                "target_bytes": arm.target_bytes,
                "exact_bytes": arm.exact_bytes,
                "maximum_rate_error_bytes": arm.maximum_rate_error_bytes,
                "role": arm.role,
            }
            for arm in arms
        ],
        "contract_checks": contract_checks,
        "summaries": summaries,
        "gate_g_c2": gate,
        "per_cloud": per_cloud_rows,
        "timing_breakdown": timing,
        "elapsed_seconds": elapsed_seconds,
    }
    metrics_path = output_dir / "defect_anomaly_metrics.json"
    _atomic_write_json(metrics_path, result)
    run_manifest.update(
        {
            "status": "complete",
            "dataset_manifest_sha256": data_protocol["manifest_sha256"],
            "scored_rows": len(per_cloud_rows),
            "per_cloud_sha256": file_sha256(
                output_dir / "defect_per_cloud.jsonl"
            ),
            "metrics_sha256": file_sha256(metrics_path),
            "timing_breakdown": timing,
        }
    )
    _atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    progress.emit("complete", 1, 1, scored_rows=len(per_cloud_rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_041_defect_anomaly_smoke.json"),
    )
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--codec-provider")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = DefectAnomalyBenchmarkConfig.from_json(args.config)
    replacements: dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "dataset_manifest": args.dataset_manifest,
        "codec_provider": args.codec_provider,
        "output_dir": args.output_dir,
    }
    for name, value in replacements.items():
        if value is not None:
            config = replace(
                config, **{name: str(value) if isinstance(value, Path) else value}
            )
    result = run_defect_anomaly_benchmark(config, device_name=args.device)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate_g_c2": result["gate_g_c2"],
                "contract_checks": result["contract_checks"],
                "metrics": str(Path(config.output_dir) / "defect_anomaly_metrics.json"),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BenchmarkCloud",
    "CodecArm",
    "DecodedCloud",
    "DefectAnomalyBenchmarkConfig",
    "KNNNormalManifoldScorer",
    "KNNScorerConfig",
    "PointCloudCodec",
    "ScoredCloud",
    "assert_matched_bytes",
    "binary_auprc",
    "binary_auroc",
    "build_diagnostic_subset_codec_arms",
    "compute_gate_g_c2",
    "load_external_anomaly_manifest",
    "run_defect_anomaly_benchmark",
    "summarize_anomaly_rows",
]
