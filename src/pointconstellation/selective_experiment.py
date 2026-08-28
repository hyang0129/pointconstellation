"""Experiment 040: heuristic selective pass-through with a frozen decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from pointconstellation.bitstream import (
    HEADER,
    MODE_ENTROPY,
    MODE_FIXED,
    SELECTIVE_HEADER,
    decode_constellation,
    encode_constellation,
    expected_payload_bytes,
    expected_selective_payload_bytes,
    expected_selective_stream_bytes,
    expected_stream_bytes,
)
from pointconstellation.codecs import (
    Tmc3RatePoint,
    run_pc_error,
    run_tmc3,
)
from pointconstellation.data import file_sha256
from pointconstellation.headroom_experiment import (
    _batch_tensors,
    _metadata,
    _search_adam_start,
    _source_scorer,
)
from pointconstellation.irregularity import (
    LocalGeometry,
    decoder_residual_score,
    deterministic_random_scores,
    local_geometry_scores,
    nearest_distances,
    select_spaced_indices,
    stratification_bins,
    stratified_error,
)
from pointconstellation.official_stability import (
    OfficialStabilityConfig,
    _load_models,
    _synchronize,
)
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.selection_baselines import SELECTION_METHODS
from pointconstellation.selective_codec import (
    Decoder,
    decode_selective_message,
    encode_selective_message,
)
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
    stability_config_matches_artifact,
)
from pointconstellation.surface_metrics import estimate_point_normals
from pointconstellation.train import select_device

SUPPORTED_SPLITS = ("validation", "ood")
SCORE_METHODS = ("curvature", "density", "decoder_residual", "boundary")
PRIMARY_SCORE = "curvature"
PRIMARY_PRESERVED_FRACTION = 0.5
PRIMARY_PAYLOAD_BUDGETS = (40, 52, 64, 78, 96, 110)


@dataclass(frozen=True)
class SelectiveExperimentConfig:
    """Validated no-training selective pass-through configuration."""

    stability_config: str = "configs/experiment_019_stability_modelnet40.json"
    stability_artifact_dir: str = "artifacts/local/experiment_019_stability_modelnet40"
    pc_error_executable: str = "artifacts/tools/mpeg-pcc-dmetric/build/Release/pc_error"
    tmc3_executable: str = "artifacts/tools/mpeg-pcc-tmc13/build/tmc3/tmc3"
    gpcc_reference_path: str | None = (
        "artifacts/local/experiment_017_modelnet40_multiseed/seed_7/"
        "gpcc_per_cloud.jsonl"
    )
    position_bits: int = 12
    timeout_seconds: float = 120.0
    payload_budgets: tuple[int, ...] = PRIMARY_PAYLOAD_BUDGETS
    coordinate_bits: int = 12
    diagnostic_q8_stream_bytes: int | None = 26
    preserved_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    score_methods: tuple[str, ...] = SCORE_METHODS
    decoder_seeds: tuple[int, ...] = (7, 17, 29, 41, 53, 67)
    adam_evaluations: int = 64
    adam_learning_rate: float = 0.03
    selection_seed: int = 20_260_840
    irregularity_neighbors: int = 16
    irregularity_chunk_size: int = 256
    minimum_spacing: float = 0.02
    recall_tolerance: float = 0.02
    normal_neighbors: int = 12
    compute_normal_consistency: bool = True
    splits: tuple[str, ...] = SUPPORTED_SPLITS
    max_clouds_per_split: int | None = None
    batch_size: int = 4
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_840
    confidence_level: float = 0.95
    gate_max_d1_degradation_percent: float = 5.0
    output_dir: str = "artifacts/local/experiment_040_selective"

    def __post_init__(self) -> None:
        if not 2 <= self.position_bits <= 24:
            raise ValueError("position_bits must be between 2 and 24")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (
            not self.payload_budgets
            or len(set(self.payload_budgets)) != len(self.payload_budgets)
            or tuple(sorted(self.payload_budgets)) != self.payload_budgets
            or min(self.payload_budgets) < 1
        ):
            raise ValueError("payload_budgets must be unique, positive, and increasing")
        if not 2 <= self.coordinate_bits <= 24:
            raise ValueError("coordinate_bits must be between 2 and 24")
        if (
            self.diagnostic_q8_stream_bytes is not None
            and self.diagnostic_q8_stream_bytes <= HEADER.size
        ):
            raise ValueError("the q=8 stream budget must exceed its header")
        if self.preserved_fractions != (0.25, 0.5, 0.75):
            raise ValueError("preserved_fractions must be the predeclared 25/50/75%")
        if self.score_methods != SCORE_METHODS:
            raise ValueError("score_methods must contain the four predeclared scores")
        if not self.decoder_seeds or len(set(self.decoder_seeds)) != len(
            self.decoder_seeds
        ):
            raise ValueError("decoder_seeds must be nonempty and unique")
        if self.adam_evaluations < 1 or self.adam_learning_rate <= 0:
            raise ValueError("Adam evaluations and learning rate must be positive")
        if not 0 <= self.selection_seed < 2**63:
            raise ValueError("selection_seed must be a nonnegative 63-bit integer")
        if self.irregularity_neighbors < 3 or self.irregularity_chunk_size < 1:
            raise ValueError("irregularity neighbors/chunk size are invalid")
        if self.minimum_spacing < 0 or self.recall_tolerance <= 0:
            raise ValueError("spacing must be nonnegative and tolerance positive")
        if self.normal_neighbors < 3:
            raise ValueError("normal_neighbors must be at least three")
        if (
            not self.splits
            or len(set(self.splits)) != len(self.splits)
            or set(self.splits) - set(SUPPORTED_SPLITS)
        ):
            raise ValueError("splits must be unique validation and/or ood entries")
        if self.max_clouds_per_split is not None and self.max_clouds_per_split < 1:
            raise ValueError("max_clouds_per_split must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")
        if self.gate_max_d1_degradation_percent < 0:
            raise ValueError("gate D1 degradation tolerance cannot be negative")

    @classmethod
    def from_json(cls, path: Path) -> SelectiveExperimentConfig:
        values = json.loads(path.read_text())
        for key in (
            "payload_budgets",
            "preserved_fractions",
            "score_methods",
            "decoder_seeds",
            "splits",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class RatePoint:
    """One predeclared fixed-width rate allocation."""

    label: str
    coordinate_bits: int
    constellation_size: int
    payload_budget_bytes: int
    fixed_payload_bytes: int
    diagnostic_uniform_only: bool = False


def _maximum_points_for_payload(payload_bytes: int, bits: int) -> int:
    count = (8 * payload_bytes) // (3 * bits)
    while count and expected_payload_bytes(count, bits) > payload_bytes:
        count -= 1
    if count < 1:
        raise ValueError("payload budget cannot hold one coordinate point")
    return count


def _rate_points(config: SelectiveExperimentConfig) -> tuple[RatePoint, ...]:
    points = []
    for budget in config.payload_budgets:
        count = _maximum_points_for_payload(budget, config.coordinate_bits)
        points.append(
            RatePoint(
                label=f"payload_{budget}_q_{config.coordinate_bits}",
                coordinate_bits=config.coordinate_bits,
                constellation_size=count,
                payload_budget_bytes=budget,
                fixed_payload_bytes=expected_payload_bytes(
                    count, config.coordinate_bits
                ),
            )
        )
    if config.diagnostic_q8_stream_bytes is not None:
        payload_budget = config.diagnostic_q8_stream_bytes - HEADER.size
        count = _maximum_points_for_payload(payload_budget, 8)
        stream_bytes = expected_stream_bytes(count, 8)
        if stream_bytes > config.diagnostic_q8_stream_bytes:
            raise RuntimeError("q=8 diagnostic exceeds its complete stream budget")
        points.append(
            RatePoint(
                label=f"stream_{config.diagnostic_q8_stream_bytes}_q_8",
                coordinate_bits=8,
                constellation_size=count,
                payload_budget_bytes=payload_budget,
                fixed_payload_bytes=expected_payload_bytes(count, 8),
                diagnostic_uniform_only=True,
            )
        )
    return tuple(points)


def _split_counts(count: int, fraction: float) -> tuple[int, int]:
    preserved = int(math.floor(count * fraction + 0.5))
    return count - preserved, preserved


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _recorded_protocol_matches(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Accept additive protocol checks while requiring every recorded value."""

    for key, value in recorded.items():
        if key not in current:
            return False
        current_value = current[key]
        if isinstance(value, Mapping):
            if not isinstance(current_value, Mapping) or not _recorded_protocol_matches(
                value, current_value
            ):
                return False
        elif value != current_value:
            return False
    return True


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _official_config(
    config: SelectiveExperimentConfig, stability: StabilityExperimentConfig
) -> OfficialStabilityConfig:
    return OfficialStabilityConfig(
        stability_config=config.stability_config,
        stability_artifact_dir=config.stability_artifact_dir,
        pc_error_executable=config.pc_error_executable,
        position_bits=config.position_bits,
        timeout_seconds=config.timeout_seconds,
        decoder_seeds=stability.decoder_seeds,
        refiner_seeds=stability.refiner_seeds,
        splits=config.splits,
        max_clouds_per_split=max(2, config.max_clouds_per_split or 2),
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
        output_dir=config.output_dir,
    )


def _selection_seed(
    config: SelectiveExperimentConfig,
    *,
    split: str,
    metadata: Mapping[str, Any],
    label: str,
) -> int:
    identity = json.dumps(
        [
            config.selection_seed,
            split,
            metadata["family"],
            metadata["model_id"],
            metadata["sample_id"],
            label,
        ],
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (
        2**63
    )


def _search_coordinates(
    decoder: nn.Module,
    source: Tensor,
    metadata: Sequence[Mapping[str, Any]],
    *,
    split: str,
    constellation_size: int,
    bits: int,
    stability: StabilityExperimentConfig,
    config: SelectiveExperimentConfig,
    output_points: int | None = None,
) -> tuple[Tensor, float]:
    _synchronize(source.device)
    started = time.perf_counter()
    initial = torch.stack(
        [
            SELECTION_METHODS["fps"](
                cloud,
                constellation_size,
                bits,
                _selection_seed(
                    config, split=split, metadata=cloud_metadata, label="adam_fps"
                ),
                None,
            )
            for cloud, cloud_metadata in zip(source, metadata, strict=True)
        ]
    )
    scorer = _source_scorer(
        decoder,
        source,
        num_output_points=output_points or stability.num_points,
        chunk_size=stability.distance_chunk_size,
    )
    result = _search_adam_start(
        scorer,
        initial,
        bits=bits,
        budget=config.adam_evaluations,
        learning_rate=config.adam_learning_rate,
    )
    _synchronize(source.device)
    if result.decoder_evaluations_per_cloud != config.adam_evaluations:
        raise RuntimeError("Adam-STE did not consume its declared evaluation budget")
    return result.coordinates, (time.perf_counter() - started) / len(source)


def _numpy_decoder(decoder: nn.Module, device: torch.device) -> Decoder:
    def decode(
        coordinates: NDArray[np.float64], output_points: int
    ) -> NDArray[np.float64]:
        tensor = torch.from_numpy(coordinates).float().unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed = decoder(tensor, num_output_points=output_points)[0]
        return reconstructed.detach().cpu().numpy().astype(np.float64)

    return decode


class FrozenExperiment040CodecContext:
    """Frozen decoder and FPS-start Adam-STE search reused by codec providers.

    Experiment 040's runner owns the torch implementation.  This small context
    lets downstream, source-only codec providers reuse that implementation
    without importing torch into their benchmark runners.
    """

    def __init__(
        self,
        config: SelectiveExperimentConfig,
        *,
        decoder_seed: int,
        device_name: str,
    ) -> None:
        stability_path = Path(config.stability_config)
        stability = StabilityExperimentConfig.from_json(stability_path)
        if decoder_seed not in config.decoder_seeds:
            raise ValueError("provider decoder seed is absent from Experiment 040")
        if decoder_seed not in stability.decoder_seeds:
            raise ValueError("provider decoder seed is absent from Experiment 019")
        if config.coordinate_bits > stability.coordinate_bits:
            raise ValueError("provider precision exceeds decoder training precision")

        artifact_dir = Path(config.stability_artifact_dir)
        metrics_path = artifact_dir / "stability_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        artifact_config = dict(metrics["config"])
        artifact_config.setdefault(
            "pointcloud_normal_neighbors", stability.pointcloud_normal_neighbors
        )
        artifact_config.setdefault(
            "verify_pointcloud_hashes", stability.verify_pointcloud_hashes
        )
        if not stability_config_matches_artifact(artifact_config, stability):
            raise RuntimeError(
                "Experiment 019 artifact config differs from checked config"
            )
        if not all(metrics["contract_checks"].values()):
            raise RuntimeError("Experiment 019 artifact has a failed contract")

        device = select_device(device_name)
        decoder, _, model_metadata = _load_models(
            stability,
            _official_config(config, stability),
            decoder_seed=decoder_seed,
            refiner_seed=None,
            device=device,
        )
        self.config = config
        self.stability = stability
        self.decoder_seed = decoder_seed
        self.device = device
        self.decoder = decoder
        self.model_metadata = model_metadata
        self._decoder_state_hash = _state_hash(decoder)
        self._decode_numpy = _numpy_decoder(decoder, device)

    @staticmethod
    def _canonical_source(
        source: NDArray[np.float32] | NDArray[np.float64],
    ) -> NDArray[np.float32]:
        values = np.asarray(source, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values):
            raise ValueError("provider source must have shape (N, 3) with N > 0")
        if not np.isfinite(values).all():
            raise ValueError("provider source must contain finite coordinates")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("provider source must lie in [-1, 1]")
        order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
        return np.ascontiguousarray(values[order])

    def assert_frozen(self) -> None:
        """Assert the sealed decoder state is unchanged."""

        if _state_hash(self.decoder) != self._decoder_state_hash:
            raise RuntimeError("frozen Experiment 019 decoder changed during encoding")

    def search(
        self,
        source: NDArray[np.float32] | NDArray[np.float64],
        *,
        constellation_size: int,
        bits: int,
        output_points: int | None = None,
    ) -> NDArray[np.float64]:
        """Run Experiment 040's source-only FPS-start Adam-STE search once."""

        values = self._canonical_source(source)
        if constellation_size > len(values):
            raise ValueError("constellation size exceeds source cardinality")
        requested_output_points = (
            len(values) if output_points is None else output_points
        )
        if requested_output_points != len(values):
            raise ValueError("provider output count must equal source cardinality")
        if requested_output_points > self.stability.num_points:
            raise ValueError("provider output count exceeds the sealed decoder maximum")
        digest = hashlib.sha256(values.astype(">f4").tobytes()).hexdigest()
        tensor = torch.from_numpy(values).unsqueeze(0).to(self.device)
        coordinates, _ = _search_coordinates(
            self.decoder,
            tensor,
            [{"family": "experiment_041", "model_id": digest, "sample_id": 0}],
            split="defect_provider",
            constellation_size=constellation_size,
            bits=bits,
            stability=self.stability,
            config=self.config,
            output_points=requested_output_points,
        )
        self.assert_frozen()
        return coordinates[0].detach().cpu().numpy().astype(np.float64)

    def decode(
        self, coordinates: NDArray[np.float64], output_points: int
    ) -> NDArray[np.float64]:
        """Decode serialized coordinates with the sealed shared decoder."""

        reconstruction = self._decode_numpy(coordinates, output_points)
        self.assert_frozen()
        return reconstruction


def _nearest_indices(
    query: NDArray[np.float64], reference: NDArray[np.float64], *, chunk_size: int
) -> NDArray[np.int64]:
    result = np.empty(len(query), dtype=np.int64)
    reference_norms = np.einsum("ij,ij->i", reference, reference)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        part = query[start:stop]
        squared = (
            np.einsum("ij,ij->i", part, part)[:, None]
            + reference_norms[None, :]
            - 2.0 * part @ reference.T
        ).clip(min=0.0)
        result[start:stop] = squared.argmin(axis=1)
    return result


def _normal_consistency(
    reconstruction: NDArray[np.float64],
    source: NDArray[np.float64],
    source_normals: NDArray[np.float64],
    *,
    neighbors: int,
    chunk_size: int,
) -> float | None:
    if len(reconstruction) < 4:
        return None
    estimated = estimate_point_normals(
        reconstruction,
        neighbors=min(neighbors, len(reconstruction) - 1),
        chunk_size=chunk_size,
    )
    indices = _nearest_indices(reconstruction, source, chunk_size=chunk_size)
    reference = source_normals[indices]
    reference /= np.linalg.norm(reference, axis=1, keepdims=True).clip(min=1e-12)
    return float(np.mean(np.abs(np.sum(estimated * reference, axis=1))))


def _geometry_metrics(
    reconstruction: NDArray[np.float64],
    source: NDArray[np.float64],
    source_normals: NDArray[np.float64],
    geometry: LocalGeometry,
    assignments: NDArray[np.int64],
    *,
    config: SelectiveExperimentConfig,
) -> dict[str, Any]:
    errors = nearest_distances(
        source, reconstruction, chunk_size=config.irregularity_chunk_size
    )
    boundary_cutoff = float(np.quantile(geometry.boundary, 0.8))
    thin_cutoff = float(np.quantile(geometry.thin_structure, 0.8))
    boundary_mask = geometry.boundary >= boundary_cutoff
    thin_mask = geometry.thin_structure >= thin_cutoff

    def recall(mask: NDArray[np.bool_]) -> float | None:
        if not mask.any():
            return None
        return float(np.mean(errors[mask] <= config.recall_tolerance))

    consistency = None
    if config.compute_normal_consistency:
        consistency = _normal_consistency(
            reconstruction,
            source,
            source_normals,
            neighbors=config.normal_neighbors,
            chunk_size=config.irregularity_chunk_size,
        )
    return {
        "target_d1_mse_proxy": float(np.mean(errors**2)),
        "target_d1_rmse_proxy": float(np.sqrt(np.mean(errors**2))),
        "tail_p95_euclidean": float(np.quantile(errors, 0.95)),
        "tail_p99_euclidean": float(np.quantile(errors, 0.99)),
        "stratified_d1": stratified_error(errors, assignments),
        "stratification_count": len(errors),
        "boundary_recall": recall(boundary_mask),
        "thin_structure_recall": recall(thin_mask),
        "normal_consistency": consistency,
        "normal_consistency_computed": config.compute_normal_consistency,
        "boundary_definition": "top source-only boundary-score quintile",
        "thin_structure_definition": "top source-only PCA-thinness quintile",
    }


def _official_metrics(
    source: NDArray[np.float64],
    reconstruction: NDArray[np.float64],
    source_normals: NDArray[np.float64],
    *,
    config: SelectiveExperimentConfig,
    scratch_root: Path,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    digest = hashlib.sha256(
        source.astype(">f4").tobytes()
        + source_normals.astype(">f4").tobytes()
        + reconstruction.astype(">f4").tobytes()
    ).hexdigest()
    if digest not in cache:
        with tempfile.TemporaryDirectory(dir=scratch_root, prefix="pc-error-") as work:
            result = run_pc_error(
                Path(config.pc_error_executable),
                source,
                reconstruction,
                source_normals,
                work_dir=Path(work),
                position_bits=config.position_bits,
                timeout_seconds=config.timeout_seconds,
            )
        if not {"d1_mse", "d2_mse"} <= result.metrics.keys():
            raise RuntimeError("pc_error result is missing official D1 or D2")
        cache[digest] = {
            **result.metrics,
            "official_metric_seconds": result.elapsed_seconds,
        }
    return cache[digest]


def _base_row(
    *,
    rate: RatePoint,
    decoder_seed: int | None,
    split: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": "040_selective_passthrough",
        "split": split,
        "decoder_seed": decoder_seed,
        **metadata,
        "rate_point": rate.label,
        "payload_budget_bytes": rate.payload_budget_bytes,
        "coordinate_bits": rate.coordinate_bits,
        "total_coordinate_points": rate.constellation_size,
        "fixed_payload_bytes": rate.fixed_payload_bytes,
        "unused_payload_budget_bytes": (
            rate.payload_budget_bytes - rate.fixed_payload_bytes
        ),
    }


def _evaluate_uniform(
    coordinates: NDArray[np.float64],
    source: NDArray[np.float64],
    source_normals: NDArray[np.float64],
    geometry: LocalGeometry,
    assignments: NDArray[np.int64],
    *,
    decoder: Decoder,
    rate: RatePoint,
    decoder_seed: int,
    split: str,
    metadata: Mapping[str, Any],
    encode_seconds: float,
    config: SelectiveExperimentConfig,
    scratch_root: Path,
    official_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    stream = encode_constellation(
        coordinates,
        bits=rate.coordinate_bits,
        mode=MODE_FIXED,
        output_points=len(source),
    )
    entropy_stream = encode_constellation(
        coordinates,
        bits=rate.coordinate_bits,
        mode=MODE_ENTROPY,
        output_points=len(source),
    )
    packet = decode_constellation(stream)
    if not hasattr(packet, "normalization_bytes"):
        raise RuntimeError("uniform arm decoded as a selective packet")
    reconstruction = np.asarray(
        decoder(packet.coordinates, packet.output_points), dtype=np.float64
    )
    decode_seconds = time.perf_counter() - started
    if len(stream) != expected_stream_bytes(
        rate.constellation_size, rate.coordinate_bits
    ):
        raise RuntimeError("uniform fixed stream differs from declared formula")
    return {
        **_base_row(
            rate=rate,
            decoder_seed=decoder_seed,
            split=split,
            metadata=metadata,
        ),
        "method": "uniform_constellation",
        "score_method": None,
        "preserved_fraction": 0.0,
        "k1": rate.constellation_size,
        "k2": 0,
        "representation_class": "free-coordinate",
        "fixed_header_bytes": HEADER.size,
        "fixed_stream_bytes": len(stream),
        "mode_1_stream_bytes": len(entropy_stream),
        "payload_bpp": 8.0 * rate.fixed_payload_bytes / len(source),
        "actual_stream_bpp": 8.0 * len(stream) / len(source),
        "mode_1_stream_bpp": 8.0 * len(entropy_stream) / len(source),
        "stream_hex": stream.hex(),
        "stream_sha256": hashlib.sha256(stream).hexdigest(),
        "serialized_round_trip_exact": encode_constellation(
            packet.coordinates,
            bits=packet.bits,
            mode=packet.mode,
            output_points=packet.output_points,
        )
        == stream,
        "preservation_error": None,
        "source_only_selection": True,
        "adam_decoder_evaluations": config.adam_evaluations,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        **_geometry_metrics(
            reconstruction,
            source,
            source_normals,
            geometry,
            assignments,
            config=config,
        ),
        **_official_metrics(
            source,
            reconstruction,
            source_normals,
            config=config,
            scratch_root=scratch_root,
            cache=official_cache,
        ),
    }


def _score_values(
    method: str,
    geometry: LocalGeometry,
    source: NDArray[np.float64],
    base_reconstruction: NDArray[np.float64],
    *,
    random_seed: int,
    chunk_size: int,
) -> NDArray[np.float64]:
    if method == "curvature":
        return geometry.curvature
    if method == "density":
        return geometry.density_deviation
    if method == "decoder_residual":
        return decoder_residual_score(
            source, base_reconstruction, chunk_size=chunk_size
        )
    if method == "boundary":
        return geometry.boundary
    if method == "random":
        return deterministic_random_scores(source, seed=random_seed)
    raise ValueError(f"unknown selective score method: {method}")


def _evaluate_selective(
    coordinates: NDArray[np.float64],
    source: NDArray[np.float64],
    source_normals: NDArray[np.float64],
    geometry: LocalGeometry,
    assignments: NDArray[np.int64],
    *,
    decoder: Decoder,
    rate: RatePoint,
    k1: int,
    k2: int,
    fraction: float,
    score_method: str,
    decoder_seed: int,
    split: str,
    metadata: Mapping[str, Any],
    search_seconds: float,
    config: SelectiveExperimentConfig,
    scratch_root: Path,
    official_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(coordinates) != k1:
        raise ValueError("selective constellation does not match K1")
    base_reconstruction = np.asarray(
        decoder(coordinates, len(source)), dtype=np.float64
    )
    random_seed = _selection_seed(
        config,
        split=split,
        metadata=metadata,
        label=f"random:{rate.label}:{fraction}",
    )
    scores = _score_values(
        score_method,
        geometry,
        source,
        base_reconstruction,
        random_seed=random_seed,
        chunk_size=config.irregularity_chunk_size,
    )
    selection_started = time.perf_counter()
    indices = select_spaced_indices(
        source, scores, k2, minimum_spacing=config.minimum_spacing
    )
    preserved = source[indices]
    stream = encode_selective_message(
        coordinates,
        preserved,
        bits=rate.coordinate_bits,
        output_points=len(source),
    )
    entropy_stream = encode_selective_message(
        coordinates,
        preserved,
        bits=rate.coordinate_bits,
        output_points=len(source),
        entropy=True,
    )
    selection_seconds = time.perf_counter() - selection_started
    decode_started = time.perf_counter()
    decoded = decode_selective_message(stream, decoder=decoder)
    decode_seconds = time.perf_counter() - decode_started
    expected_payload = expected_selective_payload_bytes(k1, k2, rate.coordinate_bits)
    expected_stream = expected_selective_stream_bytes(k1, k2, rate.coordinate_bits)
    if len(stream) != expected_stream or expected_payload != rate.fixed_payload_bytes:
        raise RuntimeError("selective stream differs from the declared fixed rate")
    return {
        **_base_row(
            rate=rate,
            decoder_seed=decoder_seed,
            split=split,
            metadata=metadata,
        ),
        "method": (
            "random_preserved_control"
            if score_method == "random"
            else "selective_passthrough"
        ),
        "score_method": score_method,
        "preserved_fraction": fraction,
        "k1": k1,
        "k2": k2,
        "representation_class": "selective-coordinate-ablation",
        "fixed_header_bytes": SELECTIVE_HEADER.size,
        "fixed_stream_bytes": len(stream),
        "mode_1_stream_bytes": len(entropy_stream),
        "payload_bpp": 8.0 * expected_payload / len(source),
        "actual_stream_bpp": 8.0 * len(stream) / len(source),
        "mode_1_stream_bpp": 8.0 * len(entropy_stream) / len(source),
        "stream_hex": stream.hex(),
        "stream_sha256": hashlib.sha256(stream).hexdigest(),
        "serialized_round_trip_exact": encode_selective_message(
            decoded.packet.constellation_coordinates,
            decoded.packet.preserved_coordinates,
            bits=decoded.packet.bits,
            output_points=decoded.packet.output_points,
        )
        == stream,
        "preservation_error": decoded.preservation_error,
        "preserved_source_quantization_max_error": float(
            np.max(np.abs(decoded.packet.preserved_coordinates - preserved))
        )
        if k2
        else 0.0,
        "minimum_selected_spacing": (
            _minimum_pair_spacing(decoded.packet.preserved_coordinates)
            if k2 > 1
            else None
        ),
        "source_only_selection": True,
        "adam_decoder_evaluations": config.adam_evaluations,
        "encode_seconds": search_seconds + selection_seconds,
        "decode_seconds": decode_seconds,
        **_geometry_metrics(
            decoded.reconstruction,
            source,
            source_normals,
            geometry,
            assignments,
            config=config,
        ),
        **_official_metrics(
            source,
            decoded.reconstruction,
            source_normals,
            config=config,
            scratch_root=scratch_root,
            cache=official_cache,
        ),
    }


def _gpcc_reference_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = [row for row in rows if row.get("method") == "gpcc_octree"]
    if not result:
        raise RuntimeError("G-PCC reference contains no geometry rows")
    return result


def _minimum_pair_spacing(points: NDArray[np.float64]) -> float:
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def _gpcc_rate_points(rows: Sequence[Mapping[str, Any]]) -> tuple[Tmc3RatePoint, ...]:
    rates: dict[str, tuple[str, ...]] = {}
    for row in rows:
        name = str(row["rate_point"])
        arguments = tuple(str(value) for value in row["encoder_args"])
        if name in rates and rates[name] != arguments:
            raise RuntimeError("G-PCC reference rate has inconsistent encoder args")
        rates[name] = arguments
    return tuple(
        Tmc3RatePoint(name=name, encoder_args=arguments)
        for name, arguments in sorted(rates.items())
    )


def _evaluate_gpcc(
    source: NDArray[np.float64],
    source_normals: NDArray[np.float64],
    geometry: LocalGeometry,
    assignments: NDArray[np.int64],
    *,
    rate: RatePoint,
    split: str,
    metadata: Mapping[str, Any],
    reference_rows: Sequence[Mapping[str, Any]],
    tmc3_executable: Path,
    scratch_root: Path,
    official_cache: dict[str, dict[str, Any]],
    frontier_cache: dict[tuple[Any, ...], list[tuple[Any, Path]]],
    config: SelectiveExperimentConfig,
) -> dict[str, Any]:
    source_digest = hashlib.sha256(source.astype(">f4").tobytes()).hexdigest()
    cache_key = (
        split,
        metadata["family"],
        metadata["model_id"],
        metadata["sample_id"],
        source_digest,
    )
    if cache_key not in frontier_cache:
        frontier = []
        for gpcc_rate in _gpcc_rate_points(reference_rows):
            work_dir = (
                scratch_root
                / "gpcc_work"
                / split
                / f"sample_{int(metadata['sample_id']):05d}"
                / gpcc_rate.name
            )
            gpcc_result = run_tmc3(
                tmc3_executable,
                source,
                rate_point=gpcc_rate,
                work_dir=work_dir,
                position_bits=config.position_bits,
                timeout_seconds=config.timeout_seconds,
            )
            frontier.append((gpcc_result, work_dir / "stream.bin"))
        frontier_cache[cache_key] = frontier
    gpcc_result, stream_path = min(
        frontier_cache[cache_key],
        key=lambda item: (
            abs(item[0].stream_breakdown.payload_bytes - rate.payload_budget_bytes),
            item[0].stream_bytes,
            item[1].parent.name,
        ),
    )
    if gpcc_result.stream_breakdown is None:
        raise RuntimeError("G-PCC result is missing its byte-exact breakdown")
    breakdown = gpcc_result.stream_breakdown
    reconstruction = gpcc_result.reconstruction.astype(np.float64)
    return {
        **_base_row(
            rate=rate,
            decoder_seed=None,
            split=split,
            metadata=metadata,
        ),
        "method": "gpcc_octree",
        "score_method": None,
        "preserved_fraction": None,
        "k1": None,
        "k2": None,
        "representation_class": "standard-geometry-codec",
        "gpcc_rate_point": stream_path.parent.name,
        "gpcc_payload_byte_delta": (
            breakdown.payload_bytes - rate.payload_budget_bytes
        ),
        "gpcc_absolute_payload_byte_delta": abs(
            breakdown.payload_bytes - rate.payload_budget_bytes
        ),
        "fixed_header_bytes": breakdown.header_bytes,
        "fixed_payload_bytes": breakdown.payload_bytes,
        "fixed_stream_bytes": breakdown.total_bytes,
        "mode_1_stream_bytes": None,
        "unused_payload_budget_bytes": (
            rate.payload_budget_bytes - breakdown.payload_bytes
        ),
        "payload_bpp": 8.0 * breakdown.payload_bytes / len(source),
        "actual_stream_bpp": 8.0 * breakdown.total_bytes / len(source),
        "mode_1_stream_bpp": None,
        "source_identity_max_error": 0.0,
        "gpcc_source_reencoded_for_experiment_040": True,
        "stream_hex": None,
        "stream_path": str(stream_path),
        "stream_sha256": file_sha256(stream_path),
        "serialized_round_trip_exact": True,
        "preservation_error": None,
        "source_only_selection": True,
        "encode_seconds": gpcc_result.encode_seconds,
        "decode_seconds": gpcc_result.decode_seconds,
        **_geometry_metrics(
            reconstruction,
            source,
            source_normals,
            geometry,
            assignments,
            config=config,
        ),
        **_official_metrics(
            source,
            reconstruction,
            source_normals,
            config=config,
            scratch_root=scratch_root,
            cache=official_cache,
        ),
    }


def _category_cloud_draw(
    categories: NDArray[np.str_], rng: np.random.Generator
) -> NDArray[np.int64]:
    unique = np.unique(categories)
    drawn_categories = unique[rng.integers(0, len(unique), size=len(unique))]
    selected = []
    for category in drawn_categories:
        choices = np.flatnonzero(categories == category)
        selected.extend(
            choices[rng.integers(0, len(choices), size=len(choices))].tolist()
        )
    return np.asarray(selected, dtype=np.int64)


def _stratified_slope(values: NDArray[np.float64]) -> float:
    rmse = np.sqrt(np.mean(values, axis=tuple(range(values.ndim - 1))))
    return float(np.polyfit(np.arange(1, 6, dtype=np.float64), rmse, 1)[0])


def _gate_comparison(
    baseline_strata: NDArray[np.float64],
    candidate_strata: NDArray[np.float64],
    baseline_d1: NDArray[np.float64],
    candidate_d1: NDArray[np.float64],
    categories: NDArray[np.str_],
    *,
    config: SelectiveExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    def effects(
        first_strata: NDArray[np.float64],
        second_strata: NDArray[np.float64],
        first_d1: NDArray[np.float64],
        second_d1: NDArray[np.float64],
    ) -> tuple[float, float]:
        flattening = _stratified_slope(first_strata) - _stratified_slope(second_strata)
        first_rmse = math.sqrt(float(first_d1.mean()))
        second_rmse = math.sqrt(float(second_d1.mean()))
        degradation = 100.0 * (second_rmse - first_rmse) / max(first_rmse, 1e-12)
        return flattening, degradation

    point_flattening, point_degradation = effects(
        baseline_strata, candidate_strata, baseline_d1, candidate_d1
    )
    rng = np.random.default_rng(seed)
    flattening = np.empty(config.bootstrap_samples, dtype=np.float64)
    degradation = np.empty(config.bootstrap_samples, dtype=np.float64)
    for draw in range(config.bootstrap_samples):
        seed_draw = rng.integers(
            0, baseline_strata.shape[0], size=baseline_strata.shape[0]
        )
        cloud_draw = _category_cloud_draw(categories, rng)
        flattening[draw], degradation[draw] = effects(
            baseline_strata[seed_draw[:, None], cloud_draw[None, :]],
            candidate_strata[seed_draw[:, None], cloud_draw[None, :]],
            baseline_d1[seed_draw[:, None], cloud_draw[None, :]],
            candidate_d1[seed_draw[:, None], cloud_draw[None, :]],
        )
    alpha = (1.0 - config.confidence_level) / 2.0
    flattening_interval = np.quantile(flattening, (alpha, 1.0 - alpha))
    degradation_interval = np.quantile(degradation, (alpha, 1.0 - alpha))
    return {
        "slope_reduction": point_flattening,
        "slope_reduction_ci_lower": float(flattening_interval[0]),
        "slope_reduction_ci_upper": float(flattening_interval[1]),
        "aggregate_d1_rmse_degradation_percent": point_degradation,
        "aggregate_d1_rmse_degradation_ci_lower_percent": float(
            degradation_interval[0]
        ),
        "aggregate_d1_rmse_degradation_ci_upper_percent": float(
            degradation_interval[1]
        ),
        "bootstrap_samples": config.bootstrap_samples,
        "confidence_level": config.confidence_level,
    }


def gate_g_c1(
    rows: Sequence[Mapping[str, Any]], config: SelectiveExperimentConfig
) -> dict[str, Any]:
    """Evaluate the predeclared curvature-50% selective flattening gate."""

    expected_budgets = set(PRIMARY_PAYLOAD_BUDGETS)
    available_budgets = {
        int(row["payload_budget_bytes"])
        for row in rows
        if row["coordinate_bits"] == 12 and row["method"] == "uniform_constellation"
    }
    evaluable = (
        len(config.decoder_seeds) >= 3
        and set(SUPPORTED_SPLITS) <= set(config.splits)
        and expected_budgets <= available_budgets
    )
    definition = (
        "curvature-score selective pass-through at the 50% split must have a "
        "positive paired-bootstrap reduction of the D1-RMSE-versus-curvature-"
        "quintile slope with no more than the configured aggregate D1 RMSE "
        "degradation at at least four of six validation payload points; the "
        "point slope reduction must also be positive at at least four OOD points"
    )
    if not evaluable:
        return {
            "gate": "G-C1",
            "definition": definition,
            "evaluable": False,
            "passes": None,
            "decision": "pending_full_six-seed_validation_and_ood_grid",
            "comparisons": [],
        }
    comparisons = []
    for split_index, split in enumerate(SUPPORTED_SPLITS):
        for budget_index, budget in enumerate(PRIMARY_PAYLOAD_BUDGETS):
            baseline_rows = [
                row
                for row in rows
                if row["split"] == split
                and row["payload_budget_bytes"] == budget
                and row["coordinate_bits"] == 12
                and row["method"] == "uniform_constellation"
            ]
            candidate_rows = [
                row
                for row in rows
                if row["split"] == split
                and row["payload_budget_bytes"] == budget
                and row["coordinate_bits"] == 12
                and row["method"] == "selective_passthrough"
                and row["score_method"] == PRIMARY_SCORE
                and row["preserved_fraction"] == PRIMARY_PRESERVED_FRACTION
            ]
            cloud_keys = sorted(
                {
                    (row["family"], row["model_id"], row["sample_id"])
                    for row in baseline_rows
                }
            )
            base_index = {
                (
                    row["decoder_seed"],
                    row["family"],
                    row["model_id"],
                    row["sample_id"],
                ): row
                for row in baseline_rows
            }
            candidate_index = {
                (
                    row["decoder_seed"],
                    row["family"],
                    row["model_id"],
                    row["sample_id"],
                ): row
                for row in candidate_rows
            }
            baseline_strata = np.asarray(
                [
                    [
                        [
                            cell["mse"]
                            for cell in base_index[(seed, *cloud)]["stratified_d1"]
                        ]
                        for cloud in cloud_keys
                    ]
                    for seed in config.decoder_seeds
                ],
                dtype=np.float64,
            )
            candidate_strata = np.asarray(
                [
                    [
                        [
                            cell["mse"]
                            for cell in candidate_index[(seed, *cloud)]["stratified_d1"]
                        ]
                        for cloud in cloud_keys
                    ]
                    for seed in config.decoder_seeds
                ],
                dtype=np.float64,
            )
            baseline_d1 = np.asarray(
                [
                    [base_index[(seed, *cloud)]["d1_mse"] for cloud in cloud_keys]
                    for seed in config.decoder_seeds
                ]
            )
            candidate_d1 = np.asarray(
                [
                    [candidate_index[(seed, *cloud)]["d1_mse"] for cloud in cloud_keys]
                    for seed in config.decoder_seeds
                ]
            )
            comparison = _gate_comparison(
                baseline_strata,
                candidate_strata,
                baseline_d1,
                candidate_d1,
                np.asarray([cloud[0] for cloud in cloud_keys]),
                config=config,
                seed=config.bootstrap_seed + 100 * split_index + budget_index,
            )
            comparison.update(
                {
                    "split": split,
                    "payload_budget_bytes": budget,
                    "passes_validation_cell": (
                        split == "validation"
                        and comparison["slope_reduction_ci_lower"] > 0
                        and comparison["aggregate_d1_rmse_degradation_ci_upper_percent"]
                        <= config.gate_max_d1_degradation_percent
                    ),
                    "positive_ood_point_slope_reduction": (
                        split == "ood" and comparison["slope_reduction"] > 0
                    ),
                }
            )
            comparisons.append(comparison)
    validation_passes = sum(bool(row["passes_validation_cell"]) for row in comparisons)
    ood_positive = sum(
        bool(row["positive_ood_point_slope_reduction"]) for row in comparisons
    )
    passes = validation_passes >= 4 and ood_positive >= 4
    return {
        "gate": "G-C1",
        "definition": definition,
        "primary_arm": {
            "score_method": PRIMARY_SCORE,
            "preserved_fraction": PRIMARY_PRESERVED_FRACTION,
        },
        "maximum_d1_rmse_degradation_percent": (config.gate_max_d1_degradation_percent),
        "evaluable": True,
        "validation_passing_rate_points": validation_passes,
        "ood_positive_rate_points": ood_positive,
        "required_rate_points": 4,
        "passes": passes,
        "decision": "continue_track_c" if passes else "stop_track_c",
        "comparisons": comparisons,
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row["split"],
            row["rate_point"],
            row["method"],
            row.get("score_method"),
            row.get("preserved_fraction"),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        result.append(
            {
                "split": key[0],
                "rate_point": key[1],
                "method": key[2],
                "score_method": key[3],
                "preserved_fraction": key[4],
                "rows": len(group),
                "payload_budget_bytes": group[0]["payload_budget_bytes"],
                "mean_fixed_stream_bytes": float(
                    np.mean([row["fixed_stream_bytes"] for row in group])
                ),
                "mean_mode_1_stream_bytes": float(
                    np.mean(
                        [
                            row["mode_1_stream_bytes"]
                            for row in group
                            if row["mode_1_stream_bytes"] is not None
                        ]
                    )
                )
                if any(row["mode_1_stream_bytes"] is not None for row in group)
                else None,
                "official_d1_rmse": math.sqrt(
                    float(np.mean([row["d1_mse"] for row in group]))
                ),
                "official_d2_rmse": math.sqrt(
                    float(np.mean([row["d2_mse"] for row in group]))
                ),
                "tail_p95_euclidean": float(
                    np.mean([row["tail_p95_euclidean"] for row in group])
                ),
                "tail_p99_euclidean": float(
                    np.mean([row["tail_p99_euclidean"] for row in group])
                ),
                "boundary_recall": float(
                    np.mean([row["boundary_recall"] for row in group])
                ),
                "thin_structure_recall": float(
                    np.mean([row["thin_structure_recall"] for row in group])
                ),
                "normal_consistency": float(
                    np.mean(
                        [
                            row["normal_consistency"]
                            for row in group
                            if row["normal_consistency"] is not None
                        ]
                    )
                )
                if any(row["normal_consistency"] is not None for row in group)
                else None,
                "encode_seconds": float(
                    np.mean([row["encode_seconds"] for row in group])
                ),
                "decode_seconds": float(
                    np.mean([row["decode_seconds"] for row in group])
                ),
            }
        )
    return result


def _paired_arm_comparisons(
    rows: Sequence[Mapping[str, Any]], config: SelectiveExperimentConfig
) -> list[dict[str, Any]]:
    """Return paired seed/category/cloud intervals against uniform K."""

    uniform = [row for row in rows if row["method"] == "uniform_constellation"]
    candidates = [
        row
        for row in rows
        if row["method"] in {"selective_passthrough", "random_preserved_control"}
    ]
    candidate_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in candidates:
        key = (
            row["split"],
            row["rate_point"],
            row["method"],
            row["score_method"],
            row["preserved_fraction"],
        )
        candidate_groups.setdefault(key, []).append(row)
    comparisons = []
    for group_index, (key, group) in enumerate(
        sorted(candidate_groups.items(), key=lambda item: str(item[0]))
    ):
        split, rate_point, method, score_method, fraction = key
        cloud_keys = sorted(
            {(row["family"], row["model_id"], row["sample_id"]) for row in group}
        )
        baseline_index = {
            (
                row["decoder_seed"],
                row["family"],
                row["model_id"],
                row["sample_id"],
            ): row
            for row in uniform
            if row["split"] == split and row["rate_point"] == rate_point
        }
        candidate_index = {
            (
                row["decoder_seed"],
                row["family"],
                row["model_id"],
                row["sample_id"],
            ): row
            for row in group
        }
        expected_keys = {
            (seed, *cloud) for seed in config.decoder_seeds for cloud in cloud_keys
        }
        if (
            set(baseline_index) != expected_keys
            or set(candidate_index) != expected_keys
        ):
            raise RuntimeError("paired selective comparison has an incomplete grid")
        categories = np.asarray([cloud[0] for cloud in cloud_keys])
        metric_arrays = {}
        for metric in ("d1_mse", "d2_mse", "tail_p99_euclidean"):
            metric_arrays[metric] = (
                np.asarray(
                    [
                        [baseline_index[(seed, *cloud)][metric] for cloud in cloud_keys]
                        for seed in config.decoder_seeds
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        [
                            candidate_index[(seed, *cloud)][metric]
                            for cloud in cloud_keys
                        ]
                        for seed in config.decoder_seeds
                    ],
                    dtype=np.float64,
                ),
            )

        def effect(
            baseline: NDArray[np.float64],
            candidate: NDArray[np.float64],
            metric: str,
        ) -> float:
            first = float(baseline.mean())
            second = float(candidate.mean())
            if metric in {"d1_mse", "d2_mse"}:
                first = math.sqrt(first)
                second = math.sqrt(second)
            return 100.0 * (first - second) / max(first, 1e-12)

        rng = np.random.default_rng(config.bootstrap_seed + 10_000 + group_index)
        draws = {
            metric: np.empty(config.bootstrap_samples, dtype=np.float64)
            for metric in metric_arrays
        }
        for draw in range(config.bootstrap_samples):
            seed_draw = rng.integers(
                0, len(config.decoder_seeds), size=len(config.decoder_seeds)
            )
            cloud_draw = _category_cloud_draw(categories, rng)
            for metric, (baseline, candidate) in metric_arrays.items():
                draws[metric][draw] = effect(
                    baseline[seed_draw[:, None], cloud_draw[None, :]],
                    candidate[seed_draw[:, None], cloud_draw[None, :]],
                    metric,
                )
        alpha = (1.0 - config.confidence_level) / 2.0
        metric_results = {}
        for metric, (baseline, candidate) in metric_arrays.items():
            lower, upper = np.quantile(draws[metric], (alpha, 1.0 - alpha))
            metric_results[metric] = {
                "relative_improvement_percent": effect(baseline, candidate, metric),
                "confidence_interval_lower_percent": float(lower),
                "confidence_interval_upper_percent": float(upper),
                "passes_positive_interval": bool(lower > 0),
            }
        comparisons.append(
            {
                "split": split,
                "rate_point": rate_point,
                "baseline_method": "uniform_constellation",
                "candidate_method": method,
                "score_method": score_method,
                "preserved_fraction": fraction,
                "decoder_seeds": len(config.decoder_seeds),
                "clouds": len(cloud_keys),
                "bootstrap_unit": (
                    "paired decoder seed and category-then-cloud resampling"
                ),
                "bootstrap_samples": config.bootstrap_samples,
                "confidence_level": config.confidence_level,
                "metrics": metric_results,
            }
        )
    return comparisons


def run_selective_experiment(
    config: SelectiveExperimentConfig, *, device_name: str | None = None
) -> dict[str, Any]:
    """Run Experiment 040 without changing any shared model parameters."""

    started = time.perf_counter()
    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    if set(config.decoder_seeds) - set(stability.decoder_seeds):
        raise ValueError("selective decoder seeds are absent from Experiment 019")
    if config.coordinate_bits > stability.coordinate_bits:
        raise ValueError("selective precision exceeds decoder training precision")
    if config.position_bits != stability.coordinate_bits:
        raise ValueError("pc_error precision must match Experiment 019")
    if config.irregularity_neighbors >= stability.num_points:
        raise ValueError("irregularity_neighbors must be smaller than source N")
    executable = Path(config.pc_error_executable)
    if not executable.is_file() or not executable.stat().st_mode & 0o111:
        raise FileNotFoundError(f"pc_error is missing or not executable: {executable}")
    tmc3_executable = Path(config.tmc3_executable)
    if config.gpcc_reference_path and (
        not tmc3_executable.is_file() or not tmc3_executable.stat().st_mode & 0o111
    ):
        raise FileNotFoundError(
            f"TMC13 is missing or not executable: {tmc3_executable}"
        )
    artifact_dir = Path(config.stability_artifact_dir)
    stability_metrics_path = artifact_dir / "stability_metrics.json"
    stability_metrics = json.loads(stability_metrics_path.read_text())
    artifact_config = dict(stability_metrics["config"])
    artifact_config.setdefault(
        "pointcloud_normal_neighbors", stability.pointcloud_normal_neighbors
    )
    artifact_config.setdefault(
        "verify_pointcloud_hashes", stability.verify_pointcloud_hashes
    )
    if not stability_config_matches_artifact(artifact_config, stability):
        raise RuntimeError("Experiment 019 artifact config differs from checked config")
    if not all(stability_metrics["contract_checks"].values()):
        raise RuntimeError("Experiment 019 artifact has a failed scientific contract")
    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    if not _recorded_protocol_matches(
        stability_metrics["data_protocol"], data_protocol
    ):
        raise RuntimeError("Experiment 019 data identity changed before Experiment 040")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = output_dir / "metric_scratch"
    scratch_root.mkdir(exist_ok=True)
    gpcc_path = Path(config.gpcc_reference_path) if config.gpcc_reference_path else None
    gpcc_rows = _gpcc_reference_rows(gpcc_path) if gpcc_path else []
    device = select_device(device_name)
    official = _official_config(config, stability)
    rates = _rate_points(config)
    manifest = {
        "experiment": "040_selective_passthrough",
        "config": _json_ready(asdict(config)),
        "stability_config_sha256": file_sha256(stability_path),
        "stability_metrics_sha256": file_sha256(stability_metrics_path),
        "pc_error_sha256": file_sha256(executable),
        "tmc3_sha256": (
            file_sha256(tmc3_executable) if config.gpcc_reference_path else None
        ),
        "gpcc_reference_sha256": file_sha256(gpcc_path) if gpcc_path else None,
        "data_protocol": data_protocol,
        "rate_points": [_json_ready(asdict(rate)) for rate in rates],
        "scientific_contract": {
            "selective_arm_is_explicit_ablation": True,
            "encoder_visibility": "source_points_only",
            "decoder_inputs": "serialized_constellation_coordinates_only",
            "preserved_points_bypass_decoder_and_are_count_delimited": True,
            "no_per_point_type_channel": True,
            "fixed_width_payload_budget_excludes_headers": True,
            "shared_decoder_model_bytes_are_separate_from_per_cloud_rate": True,
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    official_cache: dict[str, dict[str, Any]] = {}
    gpcc_frontier_cache: dict[tuple[Any, ...], list[tuple[Any, Path]]] = {}
    geometry_cache: dict[tuple[str, int], tuple[LocalGeometry, NDArray[np.int64]]] = {}
    gpcc_completed: set[tuple[str, str, int]] = set()
    model_records = []
    for decoder_seed in config.decoder_seeds:
        decoder, _, decoder_metadata = _load_models(
            stability,
            official,
            decoder_seed=decoder_seed,
            refiner_seed=None,
            device=device,
        )
        decoder_hash = _state_hash(decoder)
        decoder_checkpoint_path = (
            artifact_dir / f"decoders/seed_{decoder_seed}/stabilized.pt"
        )
        model_records.append(
            {
                "decoder_seed": decoder_seed,
                "decoder_parameter_count": sum(
                    parameter.numel() for parameter in decoder.parameters()
                ),
                "shared_decoder_checkpoint_bytes": (
                    decoder_checkpoint_path.stat().st_size
                ),
                "shared_model_cost_included_in_per_cloud_rate": False,
                **decoder_metadata,
            }
        )
        decode_numpy = _numpy_decoder(decoder, device)
        for split in config.splits:
            dataset = datasets[split]
            cloud_count = (
                len(dataset)
                if config.max_clouds_per_split is None
                else min(len(dataset), config.max_clouds_per_split)
            )
            for batch_start in range(0, cloud_count, config.batch_size):
                indices = list(
                    range(
                        batch_start,
                        min(batch_start + config.batch_size, cloud_count),
                    )
                )
                samples = [dataset[index] for index in indices]
                metadata = [
                    _metadata(sample, index)
                    for sample, index in zip(samples, indices, strict=True)
                ]
                source_tensor, _, normals_tensor = _batch_tensors(samples, device)
                source_numpy = source_tensor.detach().cpu().numpy().astype(np.float64)
                normals_numpy = normals_tensor.detach().cpu().numpy().astype(np.float64)
                searches: dict[tuple[int, int], tuple[Tensor, float]] = {}
                for bits in sorted({rate.coordinate_bits for rate in rates}):
                    sizes = {
                        rate.constellation_size
                        for rate in rates
                        if rate.coordinate_bits == bits
                    }
                    for rate in rates:
                        if rate.coordinate_bits != bits or rate.diagnostic_uniform_only:
                            continue
                        sizes.update(
                            _split_counts(rate.constellation_size, fraction)[0]
                            for fraction in config.preserved_fractions
                        )
                    for size in sorted(sizes):
                        searches[(size, bits)] = _search_coordinates(
                            decoder,
                            source_tensor,
                            metadata,
                            split=split,
                            constellation_size=size,
                            bits=bits,
                            stability=stability,
                            config=config,
                        )
                for cloud_index, (sample_index, cloud_metadata) in enumerate(
                    zip(indices, metadata, strict=True)
                ):
                    source = source_numpy[cloud_index]
                    source_normals = normals_numpy[cloud_index]
                    cache_key = (split, sample_index)
                    if cache_key not in geometry_cache:
                        geometry = local_geometry_scores(
                            source,
                            neighbors=config.irregularity_neighbors,
                            chunk_size=config.irregularity_chunk_size,
                        )
                        geometry_cache[cache_key] = (
                            geometry,
                            stratification_bins(geometry.curvature, points=source),
                        )
                    geometry, assignments = geometry_cache[cache_key]
                    for rate in rates:
                        coordinates, search_seconds = searches[
                            (rate.constellation_size, rate.coordinate_bits)
                        ]
                        rows.append(
                            _evaluate_uniform(
                                coordinates[cloud_index]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float64),
                                source,
                                source_normals,
                                geometry,
                                assignments,
                                decoder=decode_numpy,
                                rate=rate,
                                decoder_seed=decoder_seed,
                                split=split,
                                metadata=cloud_metadata,
                                encode_seconds=search_seconds,
                                config=config,
                                scratch_root=scratch_root,
                                official_cache=official_cache,
                            )
                        )
                        if rate.diagnostic_uniform_only:
                            continue
                        for fraction in config.preserved_fractions:
                            k1, k2 = _split_counts(rate.constellation_size, fraction)
                            selected, selective_search_seconds = searches[
                                (k1, rate.coordinate_bits)
                            ]
                            coordinate_values = (
                                selected[cloud_index]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float64)
                            )
                            for score_method in (*config.score_methods, "random"):
                                rows.append(
                                    _evaluate_selective(
                                        coordinate_values,
                                        source,
                                        source_normals,
                                        geometry,
                                        assignments,
                                        decoder=decode_numpy,
                                        rate=rate,
                                        k1=k1,
                                        k2=k2,
                                        fraction=fraction,
                                        score_method=score_method,
                                        decoder_seed=decoder_seed,
                                        split=split,
                                        metadata=cloud_metadata,
                                        search_seconds=selective_search_seconds,
                                        config=config,
                                        scratch_root=scratch_root,
                                        official_cache=official_cache,
                                    )
                                )
                        gpcc_key = (split, rate.label, int(cloud_metadata["sample_id"]))
                        if gpcc_path and gpcc_key not in gpcc_completed:
                            rows.append(
                                _evaluate_gpcc(
                                    source,
                                    source_normals,
                                    geometry,
                                    assignments,
                                    rate=rate,
                                    split=split,
                                    metadata=cloud_metadata,
                                    reference_rows=gpcc_rows,
                                    tmc3_executable=tmc3_executable,
                                    scratch_root=scratch_root,
                                    official_cache=official_cache,
                                    frontier_cache=gpcc_frontier_cache,
                                    config=config,
                                )
                            )
                            gpcc_completed.add(gpcc_key)
        if _state_hash(decoder) != decoder_hash:
            raise RuntimeError("frozen stabilized decoder changed during evaluation")
    gate = gate_g_c1(rows, config)
    per_cloud_path = output_dir / "selective_per_cloud.jsonl"
    _write_jsonl(per_cloud_path, rows)
    result = {
        "experiment": "040_selective_passthrough",
        "config": _json_ready(asdict(config)),
        "rows": len(rows),
        "model_records": model_records,
        "aggregate": _aggregate_rows(rows),
        "paired_comparisons": _paired_arm_comparisons(rows, config),
        "gate_g_c1": gate,
        "contract_checks": {
            "source_only_selection": all(row["source_only_selection"] for row in rows),
            "exact_selective_preservation": all(
                row["preservation_error"] in {None, 0.0} for row in rows
            ),
            "exact_stream_round_trip": all(
                row["serialized_round_trip_exact"] for row in rows
            ),
            "stratification_covers_target": all(
                sum(cell["count"] for cell in row["stratified_d1"])
                == row["stratification_count"]
                for row in rows
            ),
            "official_d1_d2_present": all(
                row.get("d1_mse") is not None and row.get("d2_mse") is not None
                for row in rows
            ),
            "frozen_decoder_hashes_unchanged": True,
            "fixed_payloads_within_budget": all(
                row["method"] == "gpcc_octree"
                or row["fixed_payload_bytes"] <= row["payload_budget_bytes"]
                for row in rows
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "per_cloud_path": str(per_cloud_path),
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError("Experiment 040 scientific contract failed")
    _write_json(output_dir / "selective_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_040_selective_smoke.json"),
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-clouds-per-split", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpcc-reference", type=Path)
    args = parser.parse_args()
    config = SelectiveExperimentConfig.from_json(args.config)
    if args.max_clouds_per_split is not None:
        config = replace(config, max_clouds_per_split=args.max_clouds_per_split)
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    if args.gpcc_reference is not None:
        config = replace(config, gpcc_reference_path=str(args.gpcc_reference))
    result = run_selective_experiment(config, device_name=args.device)
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "gate_g_c1": result["gate_g_c1"],
                "metrics": str(Path(config.output_dir) / "selective_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
