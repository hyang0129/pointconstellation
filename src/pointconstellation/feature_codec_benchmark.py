"""Benchmark a matched learned feature-latent point-cloud codec."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from pointconstellation.bitstream import NORMALIZATION, expected_stream_bytes
from pointconstellation.codecs import run_pc_error
from pointconstellation.data import MeshSurfaceDataset, file_sha256, load_mesh_manifest
from pointconstellation.feature_bitstream import (
    decode_features,
    encode_features,
    expected_feature_payload_bytes,
    expected_feature_stream_bytes,
)
from pointconstellation.losses import chamfer_squared_chunked
from pointconstellation.models.feature_codec import VariableFeatureCodec
from pointconstellation.rate_accounting import model_amortization
from pointconstellation.refiner_benchmark import paired_hierarchical_bootstrap
from pointconstellation.refiner_experiment import _state_hash
from pointconstellation.stability_experiment import (
    _calibration_candidate_is_better,
    _clone_state,
    _ema_calibration_candidate,
    _independent_feature_bootstrap,
    _membership,
    _update_ema,
)
from pointconstellation.standardized_benchmark import (
    _average_rows,
    _monotonicity,
    _peak_rss_bytes,
    _timed,
)
from pointconstellation.standardized_metrics import standardized_geometry_metrics
from pointconstellation.train import select_device, set_seed


@dataclass(frozen=True)
class FeatureCodecBenchmarkConfig:
    """Fixed-data learned feature-codec comparison configuration."""

    dataset_root: str
    dataset_manifest: str
    latent_dims: tuple[int, ...]
    matched_constellation_sizes: tuple[int, ...]
    model_seeds: tuple[int, ...] = (7, 17, 29)
    data_seed: int = 1517
    num_points: int = 2048
    train_samples: int = 512
    calibration_split: str | None = None
    calibration_samples: int = 0
    validation_samples: int = 128
    category_ood_samples: int = 32
    verify_mesh_hashes: bool = True
    batch_size: int = 4
    epochs: int = 2
    learning_rate: float = 1e-3
    ema_decay: float | None = None
    calibration_selection_start_epoch: int = 2
    feature_bits: int = 8
    coordinate_bits: int = 12
    feature_width: int = 64
    distance_chunk_size: int = 256
    sliced_directions: int = 32
    peak_distance: float = 2.0
    official_metric_executable: str | None = None
    official_metric_position_bits: int = 12
    official_metric_timeout_seconds: float = 120.0
    reference_multiseed_dir: str | None = None
    reference_stability_dir: str | None = None
    expected_manifest_sha256: str | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_814
    confidence_level: float = 0.95
    primary_constellation_size: int = 8
    output_dir: str = "artifacts/local/experiment_018_feature_codec_multiseed"

    def __post_init__(self) -> None:
        if not self.model_seeds or len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("model_seeds must be nonempty and unique")
        if (
            not self.latent_dims
            or len(self.latent_dims) != len(self.matched_constellation_sizes)
            or len(set(self.latent_dims)) != len(self.latent_dims)
            or len(set(self.matched_constellation_sizes))
            != len(self.matched_constellation_sizes)
        ):
            raise ValueError("latent and constellation rate points must align uniquely")
        if self.primary_constellation_size not in self.matched_constellation_sizes:
            raise ValueError("primary constellation size is absent from rate points")
        if self.calibration_split not in {None, "train", "calibration"}:
            raise ValueError("calibration_split must be train, calibration, or null")
        if (
            min(
                self.num_points,
                self.train_samples,
                self.validation_samples,
                self.category_ood_samples,
                self.batch_size,
                self.epochs,
                self.distance_chunk_size,
            )
            < 1
        ):
            raise ValueError("sample, batch, epoch, and point counts must be positive")
        if self.calibration_samples < 0:
            raise ValueError("calibration_samples cannot be negative")
        if self.ema_decay is None:
            if self.calibration_split is not None or self.calibration_samples:
                raise ValueError("calibration selection requires ema_decay")
        else:
            if not 0.0 < self.ema_decay < 1.0:
                raise ValueError("ema_decay must be in (0, 1)")
            if self.calibration_split is None or self.calibration_samples < 1:
                raise ValueError("EMA selection requires a nonempty calibration split")
            if not 1 <= self.calibration_selection_start_epoch <= self.epochs:
                raise ValueError(
                    "calibration_selection_start_epoch must be within training"
                )
        if self.learning_rate <= 0 or self.peak_distance <= 0:
            raise ValueError("learning rate and peak distance must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")
        for latent_dim, constellation_size in zip(
            self.latent_dims, self.matched_constellation_sizes, strict=True
        ):
            feature_bytes = expected_feature_stream_bytes(latent_dim, self.feature_bits)
            constellation_bytes = expected_stream_bytes(
                constellation_size, self.coordinate_bits
            )
            if feature_bytes != constellation_bytes:
                raise ValueError(
                    "feature and constellation rate points must have equal stream bytes"
                )

    @classmethod
    def from_json(cls, path: Path) -> FeatureCodecBenchmarkConfig:
        values = json.loads(path.read_text())
        for key in (
            "latent_dims",
            "matched_constellation_sizes",
            "model_seeds",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


FeatureDatasetMap = dict[str, Dataset[dict[str, Any]]]


def _mesh_dataset(
    config: FeatureCodecBenchmarkConfig, split: str
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


def _datasets(config: FeatureCodecBenchmarkConfig) -> FeatureDatasetMap:
    train = _mesh_dataset(config, "train")
    datasets: FeatureDatasetMap
    if config.calibration_split == "train":
        expected_train = config.train_samples + config.calibration_samples
        if len(train) != expected_train:
            raise ValueError(
                f"manifest train has {len(train)} records; expected {expected_train} "
                "for the declared train/calibration holdout"
            )
        datasets = {
            "train": Subset(train, range(config.train_samples)),
            "calibration": Subset(train, range(config.train_samples, expected_train)),
        }
    else:
        if len(train) != config.train_samples:
            raise ValueError(
                f"manifest train has {len(train)} records; config expects "
                f"{config.train_samples}"
            )
        datasets = {"train": train}
        if config.calibration_split == "calibration":
            calibration = _mesh_dataset(config, "calibration")
            if len(calibration) != config.calibration_samples:
                raise ValueError(
                    f"manifest calibration has {len(calibration)} records; config "
                    f"expects {config.calibration_samples}"
                )
            datasets["calibration"] = calibration

    for split, count in (
        ("validation", config.validation_samples),
        ("category_ood", config.category_ood_samples),
    ):
        dataset = _mesh_dataset(config, split)
        if len(dataset) != count:
            raise ValueError(
                f"manifest split {split} has {len(dataset)} records; "
                f"config expects {count}"
            )
        datasets[split] = dataset
    return datasets


def _data_protocol(
    config: FeatureCodecBenchmarkConfig, datasets: FeatureDatasetMap
) -> dict[str, Any]:
    partitions = {name: _membership(dataset) for name, dataset in datasets.items()}
    record_sets = {
        name: set(partition["records"]) for name, partition in partitions.items()
    }
    pairwise_disjoint = {
        f"{left}_vs_{right}": not bool(record_sets[left] & record_sets[right])
        for index, left in enumerate(partitions)
        for right in tuple(partitions)[index + 1 :]
    }
    manifest_path = Path(config.dataset_manifest)
    manifest_sha256 = file_sha256(manifest_path)
    if (
        config.expected_manifest_sha256 is not None
        and manifest_sha256 != config.expected_manifest_sha256
    ):
        raise ValueError("dataset manifest hash differs from expected_manifest_sha256")
    return {
        "dataset": load_mesh_manifest(manifest_path)["dataset"],
        "manifest_sha256": manifest_sha256,
        "data_seed": config.data_seed,
        "calibration_source": (
            None
            if config.calibration_split is None
            else (
                "held_out_train_records"
                if config.calibration_split == "train"
                else "explicit_calibration_manifest_split"
            )
        ),
        "calibration_forbidden_splits": ["validation", "category_ood"],
        "partitions": partitions,
        "pairwise_disjoint": pairwise_disjoint,
        "all_partitions_pairwise_disjoint": all(pairwise_disjoint.values()),
    }


def _parameter_count(codec: VariableFeatureCodec) -> int:
    return sum(parameter.numel() for parameter in codec.parameters())


def _calibration_rmse(
    codec: nn.Module,
    dataset: Dataset[dict[str, Any]],
    *,
    config: FeatureCodecBenchmarkConfig,
    device: torch.device,
) -> float:
    """Measure source-visible calibration Chamfer at the primary byte rate."""

    latent_dim = dict(
        zip(config.matched_constellation_sizes, config.latent_dims, strict=True)
    )[config.primary_constellation_size]
    codec.eval()
    total = 0.0
    count = 0
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for batch in loader:
            source = batch["source_points"].to(device)
            reconstruction, _ = codec(source, latent_dim)
            loss = chamfer_squared_chunked(
                reconstruction,
                source,
                chunk_size=config.distance_chunk_size,
            )
            total += float(loss.item()) * len(source)
            count += len(source)
    return math.sqrt(total / count)


def _train_seed(
    config: FeatureCodecBenchmarkConfig,
    *,
    training_dataset: Dataset[dict[str, Any]],
    calibration_dataset: Dataset[dict[str, Any]] | None,
    model_seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[VariableFeatureCodec, dict[str, Any]]:
    set_seed(model_seed)
    codec = VariableFeatureCodec(
        config.num_points,
        max(config.latent_dims),
        bits=config.feature_bits,
        feature_width=config.feature_width,
    ).to(device)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=config.learning_rate)
    loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(model_seed),
    )
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    ema = _clone_state(codec) if config.ema_decay is not None else None
    candidates: list[dict[str, Any]] = []
    selected_state: dict[str, Tensor] | None = None
    selected_candidate: dict[str, Any] | None = None
    update_count = 0
    for epoch in range(1, config.epochs + 1):
        total = 0.0
        examples = 0
        codec.train()
        for batch_index, batch in enumerate(loader):
            latent_dim = config.latent_dims[
                ((epoch - 1) * len(loader) + batch_index) % len(config.latent_dims)
            ]
            points = batch["source_points"].to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = codec(points, latent_dim)
            loss = chamfer_squared_chunked(
                reconstruction,
                points,
                chunk_size=config.distance_chunk_size,
            )
            loss.backward()
            optimizer.step()
            if ema is not None:
                assert config.ema_decay is not None
                _update_ema(ema, codec, config.ema_decay)
            total += float(loss.item()) * len(points)
            examples += len(points)
            update_count += 1
        record: dict[str, Any] = {
            "epoch": epoch,
            "training_chamfer_rmse": math.sqrt(total / examples),
        }
        if ema is not None and epoch >= config.calibration_selection_start_epoch:
            if calibration_dataset is None:
                raise RuntimeError("EMA selection has no calibration dataset")
            candidate_state, candidate = _ema_calibration_candidate(
                codec,
                ema,
                epoch=epoch,
                calibration_score=lambda candidate_codec: _calibration_rmse(
                    candidate_codec,
                    calibration_dataset,
                    config=config,
                    device=device,
                ),
            )
            candidates.append(candidate)
            record["ema_calibration_chamfer_rmse"] = candidate[
                "calibration_chamfer_rmse"
            ]
            if _calibration_candidate_is_better(candidate, selected_candidate):
                selected_candidate = candidate
                selected_state = candidate_state
        history.append(record)
        print(
            json.dumps(
                {
                    "stage": "feature_codec",
                    "model_seed": model_seed,
                    **record,
                }
            ),
            flush=True,
        )

    if ema is None:
        selected_state = _clone_state(codec)
        selected_candidate = {
            "epoch": config.epochs,
            "kind": "raw_final_epoch",
            "calibration_chamfer_rmse": None,
            "state_hash": _state_hash(codec),
        }
    elif selected_state is None or selected_candidate is None:
        raise RuntimeError("no EMA calibration candidate was eligible")
    codec.load_state_dict(selected_state)
    selected_hash = _state_hash(codec)
    selection = {**selected_candidate, "selected_state_hash": selected_hash}
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(codec.encoder.state_dict(), output_dir / "encoder.pt")
    decoder_files = {}
    decoder_state = codec.decoder.state_dict()
    torch.save(decoder_state, output_dir / "decoder.pt")
    for precision, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        path = output_dir / f"decoder_state_dict_{precision}.pt"
        torch.save(
            {
                name: (
                    value.detach().cpu().to(dtype)
                    if value.is_floating_point()
                    else value.detach().cpu()
                )
                for name, value in decoder_state.items()
            },
            path,
        )
        decoder_files[precision] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    selection_record = {
        "model_seed": model_seed,
        "selection_metric": (
            "source-visible aggregate Chamfer RMSE at the primary matched byte rate"
            if ema is not None
            else None
        ),
        "primary_constellation_size": config.primary_constellation_size,
        "primary_latent_dim": dict(
            zip(config.matched_constellation_sizes, config.latent_dims, strict=True)
        )[config.primary_constellation_size],
        "expected_stream_bytes": expected_stream_bytes(
            config.primary_constellation_size,
            config.coordinate_bits,
            normalization=True,
        ),
        "ema_decay": config.ema_decay,
        "calibration_selection_start_epoch": (
            config.calibration_selection_start_epoch if ema is not None else None
        ),
        "calibration_partition_sha256": (
            _membership(calibration_dataset)["sha256"]
            if calibration_dataset is not None
            else None
        ),
        "candidates": candidates,
        "selection": selection,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection_record, indent=2) + "\n"
    )
    return codec.eval(), {
        "history": history,
        "optimizer_updates": update_count,
        "training_elapsed_seconds": time.perf_counter() - started,
        "parameters": _parameter_count(codec),
        "state_hash": selected_hash,
        "ema_decay": config.ema_decay,
        "calibration_candidates": candidates,
        "selection": selection,
        "selection_record": str(output_dir / "selection.json"),
        "encoder_checkpoint_bytes": (output_dir / "encoder.pt").stat().st_size,
        "decoder_checkpoint_bytes": (output_dir / "decoder.pt").stat().st_size,
        "decoder_state_dicts": decoder_files,
    }


def _evaluate_seed(
    config: FeatureCodecBenchmarkConfig,
    *,
    datasets: FeatureDatasetMap,
    codec: VariableFeatureCodec,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for split in ("validation", "category_ood"):
            for sample_id in range(len(datasets[split])):
                sample = datasets[split].sample(sample_id)
                source_cpu = torch.from_numpy(sample.source_points)
                normals_cpu = torch.from_numpy(sample.source_normals)
                fresh_cpu = torch.from_numpy(sample.target_points)
                fresh_normals_cpu = torch.from_numpy(sample.target_normals)
                original_source_cpu = torch.from_numpy(sample.original_source_points)
                source = source_cpu.to(device)[None]
                for latent_dim, constellation_size in zip(
                    config.latent_dims,
                    config.matched_constellation_sizes,
                    strict=True,
                ):
                    features, encoder_seconds = _timed(
                        device,
                        lambda source=source, latent_dim=latent_dim: codec.encoder(
                            source, latent_dim
                        ),
                    )
                    serialization_started = time.perf_counter()
                    stream = encode_features(
                        features[0].cpu().numpy(),
                        bits=config.feature_bits,
                        output_points=config.num_points,
                        normalization_center=sample.normalization_center,
                        normalization_scale=sample.normalization_scale,
                    )
                    bitstream_encode_seconds = (
                        time.perf_counter() - serialization_started
                    )

                    def decode(
                        stream: bytes = stream,
                        dtype: torch.dtype = source.dtype,
                    ) -> torch.Tensor:
                        packet = decode_features(stream)
                        decoded = torch.from_numpy(packet.features).to(
                            device=device, dtype=dtype
                        )[None]
                        return codec.decoder(
                            decoded, num_output_points=packet.output_points
                        )

                    reconstruction, decode_seconds = _timed(device, decode)
                    packet = decode_features(stream)
                    center = torch.from_numpy(packet.normalization_center).to(
                        device=device, dtype=reconstruction.dtype
                    )
                    original_reconstruction = (
                        reconstruction * packet.normalization_scale + center
                    )
                    metrics = standardized_geometry_metrics(
                        reconstruction[0].cpu(),
                        source_cpu,
                        normals_cpu,
                        chunk_size=config.distance_chunk_size,
                        sliced_directions=config.sliced_directions,
                        peak_distance=config.peak_distance,
                    )
                    fresh_metrics = standardized_geometry_metrics(
                        reconstruction[0].cpu(),
                        fresh_cpu,
                        fresh_normals_cpu,
                        chunk_size=config.distance_chunk_size,
                        sliced_directions=config.sliced_directions,
                        peak_distance=config.peak_distance,
                    )
                    metrics.update(
                        {
                            f"fresh_{name}": value
                            for name, value in fresh_metrics.items()
                        }
                    )
                    if config.official_metric_executable is not None:
                        official = run_pc_error(
                            Path(config.official_metric_executable),
                            source_cpu.numpy(),
                            reconstruction[0].cpu().numpy(),
                            normals_cpu.numpy(),
                            work_dir=(
                                output_dir
                                / "official_metric_work"
                                / split
                                / f"sample_{sample_id:05d}"
                                / f"d_{latent_dim:04d}"
                            ),
                            position_bits=config.official_metric_position_bits,
                            timeout_seconds=config.official_metric_timeout_seconds,
                        )
                        metrics.update(
                            {
                                f"official_{name}": value
                                for name, value in official.metrics.items()
                            }
                        )
                        metrics["official_elapsed_seconds"] = official.elapsed_seconds
                        original_official = run_pc_error(
                            Path(config.official_metric_executable),
                            original_source_cpu.numpy(),
                            original_reconstruction[0].cpu().numpy(),
                            normals_cpu.numpy(),
                            work_dir=(
                                output_dir
                                / "official_metric_original_frame_work"
                                / split
                                / f"sample_{sample_id:05d}"
                                / f"d_{latent_dim:04d}"
                            ),
                            position_bits=config.official_metric_position_bits,
                            timeout_seconds=config.official_metric_timeout_seconds,
                            normalization_center=sample.normalization_center,
                            normalization_scale=sample.normalization_scale,
                        )
                        metrics.update(
                            {
                                f"original_frame_official_{name}": value
                                for name, value in original_official.metrics.items()
                            }
                        )
                        metrics["original_frame_official_elapsed_seconds"] = (
                            original_official.elapsed_seconds
                        )
                    rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "family": sample.category,
                            "model_id": sample.model_id,
                            "method": "feature_latent",
                            "constellation_size": constellation_size,
                            "latent_dim": latent_dim,
                            "feature_bits": config.feature_bits,
                            "payload_bits": packet.payload_bits,
                            "nominal_payload_bpp": packet.payload_bits
                            / config.num_points,
                            "header_bytes": packet.header_bytes,
                            "payload_bytes": packet.payload_bytes,
                            "normalization_bytes": packet.normalization_bytes,
                            "payload_bpp": packet.payload_bytes * 8 / config.num_points,
                            "stream_bytes": packet.stream_bytes,
                            "actual_stream_bpp": packet.stream_bytes
                            * 8
                            / config.num_points,
                            "encoder_inference_seconds": encoder_seconds,
                            "bitstream_encode_seconds": bitstream_encode_seconds,
                            "encode_seconds": encoder_seconds
                            + bitstream_encode_seconds,
                            "decode_seconds": decode_seconds,
                            **metrics,
                        }
                    )
    return rows, time.perf_counter() - started


def _index_rows(
    rows: list[dict[str, Any]],
    *,
    method: str,
    split: str,
    constellation_size: int,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    selected = {}
    for row in rows:
        if (
            row["method"] == method
            and row["split"] == split
            and row["constellation_size"] == constellation_size
        ):
            key = (row["family"], row["model_id"], row["sample_id"])
            if key in selected:
                raise ValueError(f"duplicate benchmark row: {key}")
            selected[key] = row
    if not selected:
        raise ValueError("comparison has no matching rows")
    return selected


def _reference_comparisons(
    config: FeatureCodecBenchmarkConfig,
    seed_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if config.reference_multiseed_dir is None:
        return []
    reference_dir = Path(config.reference_multiseed_dir)
    reference_results = [
        json.loads(
            (
                reference_dir / f"seed_{model_seed}" / "benchmark_metrics.json"
            ).read_text()
        )
        for model_seed in config.model_seeds
    ]
    comparisons = []
    for split_index, split in enumerate(("validation", "category_ood")):
        for size_index, constellation_size in enumerate(
            config.matched_constellation_sizes
        ):
            feature_indices = [
                _index_rows(
                    result["per_cloud"],
                    method="feature_latent",
                    split=split,
                    constellation_size=constellation_size,
                )
                for result in seed_results
            ]
            keys = tuple(sorted(feature_indices[0]))
            for method_index, method in enumerate(("free", "strict_subset")):
                reference_indices = [
                    _index_rows(
                        result["per_cloud"],
                        method=method,
                        split=split,
                        constellation_size=constellation_size,
                    )
                    for result in reference_results
                ]
                if any(tuple(sorted(index)) != keys for index in reference_indices):
                    raise RuntimeError("feature and constellation clouds do not align")
                for feature_index, reference_index in zip(
                    feature_indices, reference_indices, strict=True
                ):
                    if any(
                        feature_index[key]["stream_bytes"]
                        != reference_index[key]["stream_bytes"]
                        + (
                            0
                            if "normalization_bytes" in reference_index[key]
                            else NORMALIZATION.size
                        )
                        for key in keys
                    ):
                        raise RuntimeError(
                            "matched learned methods differ in actual rate"
                        )
                metrics = {}
                fields = [
                    "chamfer_mse",
                    "fresh_chamfer_mse",
                    "d1_mse_proxy",
                    "d2_mse_proxy",
                ]
                if "official_d1_mse" in feature_indices[0][keys[0]]:
                    fields.extend(("official_d1_mse", "official_d2_mse"))
                for metric_index, field in enumerate(fields):
                    feature = np.asarray(
                        [
                            [index[key][field] for key in keys]
                            for index in feature_indices
                        ]
                    )
                    constellation = np.asarray(
                        [
                            [index[key][field] for key in keys]
                            for index in reference_indices
                        ]
                    )
                    statistics = paired_hierarchical_bootstrap(
                        feature,
                        constellation,
                        samples=config.bootstrap_samples,
                        confidence_level=config.confidence_level,
                        seed=(
                            config.bootstrap_seed
                            + 10_000 * split_index
                            + 1_000 * size_index
                            + 100 * method_index
                            + metric_index
                        ),
                    )
                    per_seed = []
                    for model_seed, baseline, candidate in zip(
                        config.model_seeds, feature, constellation, strict=True
                    ):
                        baseline_rmse = math.sqrt(float(baseline.mean()))
                        candidate_rmse = math.sqrt(float(candidate.mean()))
                        per_seed.append(
                            {
                                "model_seed": model_seed,
                                "constellation_relative_rmse_improvement_percent": (
                                    100.0
                                    * (baseline_rmse - candidate_rmse)
                                    / max(baseline_rmse, 1e-12)
                                ),
                                "constellation_cloud_wins": int(
                                    np.count_nonzero(candidate < baseline)
                                ),
                                "clouds": len(keys),
                            }
                        )
                    metrics[field] = {**statistics, "per_seed": per_seed}
                comparisons.append(
                    {
                        "split": split,
                        "constellation_size": constellation_size,
                        "actual_stream_bpp": feature_indices[0][keys[0]][
                            "actual_stream_bpp"
                        ],
                        "payload_bpp": feature_indices[0][keys[0]].get(
                            "payload_bpp",
                            expected_feature_payload_bytes(
                                feature_indices[0][keys[0]]["latent_dim"],
                                config.feature_bits,
                            )
                            * 8
                            / config.num_points,
                        ),
                        "feature_latent_dim": feature_indices[0][keys[0]]["latent_dim"],
                        "constellation_method": method,
                        "positive_improvement_means_constellation_is_better": True,
                        "metrics": metrics,
                    }
                )
    return comparisons


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _stability_reference_comparison(
    config: FeatureCodecBenchmarkConfig,
    seed_results: list[dict[str, Any]],
    data_protocol: dict[str, Any],
) -> dict[str, Any]:
    """Compare feature seeds with all stabilized Experiment 019 factorial cells."""

    if config.reference_stability_dir is None:
        return {"status": "not_configured", "representation_gate_passes": None}
    root = Path(config.reference_stability_dir)
    metrics_path = root / "stability_metrics.json"
    rows_path = root / "per_cloud.jsonl"
    if not metrics_path.is_file() or not rows_path.is_file():
        return {
            "status": "configured_reference_missing",
            "reference_dir": str(root),
            "representation_gate_passes": None,
        }

    reference = json.loads(metrics_path.read_text())
    reference_config = reference["config"]
    reference_protocol = reference["data_protocol"]
    protocol_checks = {
        "data_seed_matches": reference_config["data_seed"] == config.data_seed,
        "manifest_matches": (
            reference_protocol["manifest_sha256"] == data_protocol["manifest_sha256"]
        ),
        "num_points_matches": reference_config["num_points"] == config.num_points,
        "training_samples_match": (
            reference_config["train_samples"] == config.train_samples
        ),
        "calibration_samples_match": (
            reference_config["calibration_samples"] == config.calibration_samples
        ),
        "validation_samples_match": (
            reference_config["validation_samples"] == config.validation_samples
        ),
        "ood_samples_match": (
            reference_config["ood_samples"] == config.category_ood_samples
        ),
        "batch_size_matches": reference_config["batch_size"] == config.batch_size,
        "rate_curriculum_matches": (
            tuple(reference_config["training_constellation_sizes"])
            == config.matched_constellation_sizes
        ),
        "primary_constellation_size_matches": (
            reference_config["constellation_size"] == config.primary_constellation_size
        ),
        "coordinate_bits_match": (
            reference_config["coordinate_bits"] == config.coordinate_bits
        ),
        "stabilized_epochs_match": (
            reference_config["stabilized_decoder_epochs"] == config.epochs
        ),
        "ema_decay_matches": reference_config["ema_decay"] == config.ema_decay,
        "selection_start_matches": (
            reference_config["baseline_decoder_epochs"] + 1
            == config.calibration_selection_start_epoch
        ),
        "train_partition_matches": (
            reference_protocol["partitions"]["train"]["sha256"]
            == data_protocol["partitions"]["train"]["sha256"]
        ),
        "calibration_partition_matches": (
            reference_protocol["partitions"]["calibration"]["sha256"]
            == data_protocol["partitions"]["calibration"]["sha256"]
        ),
        "validation_partition_matches": (
            reference_protocol["partitions"]["validation"]["sha256"]
            == data_protocol["partitions"]["validation"]["sha256"]
        ),
        "ood_partition_matches": (
            reference_protocol["partitions"]["ood"]["sha256"]
            == data_protocol["partitions"]["category_ood"]["sha256"]
        ),
        "stability_factorial_complete": reference["factorial"]["complete"],
    }
    if not all(protocol_checks.values()):
        failed = sorted(name for name, passes in protocol_checks.items() if not passes)
        raise ValueError(
            "stability reference differs from the equal protocol: " + ", ".join(failed)
        )

    coordinate_rows = _read_jsonl(rows_path)
    decoder_seeds = tuple(reference_config["decoder_seeds"])
    refiner_seeds = tuple(reference_config["refiner_seeds"])
    expected_bytes = expected_stream_bytes(
        config.primary_constellation_size,
        config.coordinate_bits,
        normalization=True,
    )

    def normalized_total_bytes(row: dict[str, Any]) -> int:
        return int(row["stream_bytes"]) + (
            0 if "normalization_bytes" in row else NORMALIZATION.size
        )

    comparisons = []
    split_names = {"validation": "validation", "ood": "category_ood"}
    for split_index, (coordinate_split, feature_split) in enumerate(
        split_names.items()
    ):
        selected_coordinates = [
            row
            for row in coordinate_rows
            if row["arm"] == "stabilized"
            and row["split"] == coordinate_split
            and row["method"] == "refiner"
        ]
        if any(
            normalized_total_bytes(row) != expected_bytes
            for row in selected_coordinates
        ):
            raise ValueError("stability reference stream size is not rate matched")
        first_cell = [
            row
            for row in selected_coordinates
            if row["decoder_seed"] == decoder_seeds[0]
            and row["refiner_seed"] == refiner_seeds[0]
        ]
        cloud_keys = [(row["family"], row["model_id"]) for row in first_cell]
        if not cloud_keys or len(cloud_keys) != len(set(cloud_keys)):
            raise RuntimeError("stability reference has invalid cloud identities")
        families = np.asarray([key[0] for key in cloud_keys], dtype=object)
        coordinate_index = {
            (
                row["decoder_seed"],
                row["refiner_seed"],
                row["family"],
                row["model_id"],
            ): row
            for row in selected_coordinates
        }
        expected_coordinate_rows = (
            len(decoder_seeds) * len(refiner_seeds) * len(cloud_keys)
        )
        if len(coordinate_index) != expected_coordinate_rows:
            raise RuntimeError("stability reference has incomplete factorial cells")

        feature_indices = []
        for result in seed_results:
            selected_features = [
                row
                for row in result["per_cloud"]
                if row["split"] == feature_split
                and row["method"] == "feature_latent"
                and row["constellation_size"] == config.primary_constellation_size
            ]
            if any(
                normalized_total_bytes(row) != expected_bytes
                for row in selected_features
            ):
                raise ValueError("feature stream size is not rate matched")
            indexed = {
                (row["family"], row["model_id"]): row for row in selected_features
            }
            if set(indexed) != set(cloud_keys):
                raise ValueError("feature and stability cloud identities do not align")
            feature_indices.append(indexed)

        for metric_index, metric in enumerate(("chamfer_mse", "fresh_chamfer_mse")):
            coordinate = np.asarray(
                [
                    [
                        [
                            coordinate_index[(decoder_seed, refiner_seed, *key)][metric]
                            for key in cloud_keys
                        ]
                        for refiner_seed in refiner_seeds
                    ]
                    for decoder_seed in decoder_seeds
                ],
                dtype=np.float64,
            )
            feature = np.asarray(
                [
                    [feature_index[key][metric] for key in cloud_keys]
                    for feature_index in feature_indices
                ],
                dtype=np.float64,
            )
            comparisons.append(
                {
                    "split": coordinate_split,
                    "metric": metric,
                    "stream_bytes": expected_bytes,
                    "constellation_size": config.primary_constellation_size,
                    **_independent_feature_bootstrap(
                        coordinate,
                        feature,
                        families,
                        config=config,
                        seed_offset=1000 * split_index + metric_index,
                    ),
                }
            )

    primary = next(
        row
        for row in comparisons
        if row["split"] == "validation" and row["metric"] == "chamfer_mse"
    )
    gate_passes = primary["confidence_interval_lower_percent"] > 0.0
    return {
        "status": "complete",
        "reference_dir": str(root),
        "feature_seeds": list(config.model_seeds),
        "coordinate_decoder_seeds": list(decoder_seeds),
        "coordinate_refiner_seeds": list(refiner_seeds),
        "protocol_checks": protocol_checks,
        "resampling": (
            "independent coordinate-decoder, coordinate-refiner, and feature-seed "
            "factors with paired hierarchical category/cloud draws"
        ),
        "comparisons": comparisons,
        "primary": primary,
        "representation_gate_passes": gate_passes,
        "claim_if_gate_fails": "competitive with a byte-matched feature codec",
    }


def run_feature_codec_benchmark(
    config: FeatureCodecBenchmarkConfig,
    *,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train independent feature codecs and compare exact matched-rate streams."""

    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multiseed_metrics_path = output_dir / "multiseed_metrics.json"
    previous_elapsed_seconds = None
    if resume and multiseed_metrics_path.exists():
        previous_elapsed_seconds = json.loads(multiseed_metrics_path.read_text()).get(
            "elapsed_seconds"
        )
    datasets = _datasets(config)
    data_protocol = _data_protocol(config, datasets)
    if not data_protocol["all_partitions_pairwise_disjoint"]:
        raise RuntimeError("training/calibration/evaluation memberships overlap")
    started = time.perf_counter()
    seed_results = []
    for model_seed in config.model_seeds:
        seed_dir = output_dir / f"seed_{model_seed}"
        seed_metrics_path = seed_dir / "benchmark_metrics.json"
        if resume and seed_metrics_path.exists():
            seed_result = json.loads(seed_metrics_path.read_text())
            if (
                seed_result.get("model_seed") != model_seed
                or seed_result.get("data_seed") != config.data_seed
            ):
                raise RuntimeError("existing feature-codec seed does not match config")
        else:
            codec, model = _train_seed(
                config,
                training_dataset=datasets["train"],
                calibration_dataset=datasets.get("calibration"),
                model_seed=model_seed,
                device=device,
                output_dir=seed_dir / "model",
            )
            rows, evaluation_seconds = _evaluate_seed(
                config,
                datasets=datasets,
                codec=codec,
                device=device,
                output_dir=seed_dir,
            )
            seed_result = {
                "model_seed": model_seed,
                "data_seed": config.data_seed,
                "model": model,
                "summary": [
                    {
                        **row,
                        **model_amortization(
                            row["stream_bytes"],
                            config.num_points,
                            {
                                precision: record["bytes"]
                                for precision, record in model[
                                    "decoder_state_dicts"
                                ].items()
                            },
                        ),
                    }
                    for row in _average_rows(rows)
                ],
                "monotonicity": _monotonicity(_average_rows(rows)),
                "per_cloud": rows,
                "evaluation_elapsed_seconds": evaluation_seconds,
            }
            seed_metrics_path.write_text(json.dumps(seed_result, indent=2) + "\n")
            with (seed_dir / "per_cloud.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
        seed_results.append(seed_result)
    hashes = [result["model"]["state_hash"] for result in seed_results]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("independent feature-codec seeds produced duplicate hashes")
    comparisons = _reference_comparisons(config, seed_results)
    stability_comparison = _stability_reference_comparison(
        config, seed_results, data_protocol
    )
    measured_component_seconds = sum(
        result["model"]["training_elapsed_seconds"]
        + result["evaluation_elapsed_seconds"]
        for result in seed_results
    )
    primary = [
        row
        for row in comparisons
        if row["constellation_size"] == config.primary_constellation_size
        and row["constellation_method"] == "free"
    ]
    primary_passes = bool(primary) and all(
        row["metrics"]["chamfer_mse"]["confidence_interval_lower_percent"] > 0
        and all(
            seed["constellation_relative_rmse_improvement_percent"] > 0
            for seed in row["metrics"]["chamfer_mse"]["per_seed"]
        )
        for row in primary
    )
    if stability_comparison["status"] == "complete":
        primary_gate = {
            "constellation_size": config.primary_constellation_size,
            "stream_bytes": expected_stream_bytes(
                config.primary_constellation_size, config.coordinate_bits
            ),
            "normalization_bytes": NORMALIZATION.size,
            "total_stream_bytes": expected_stream_bytes(
                config.primary_constellation_size,
                config.coordinate_bits,
                normalization=True,
            ),
            "metric": "validation source-cloud Chamfer RMSE",
            "criterion": (
                "stabilized constellation relative improvement confidence interval "
                "excludes zero on the positive side"
            ),
            "passes": stability_comparison["representation_gate_passes"],
            "claim_if_gate_fails": "competitive with a byte-matched feature codec",
        }
    else:
        primary_gate = {
            "constellation_size": config.primary_constellation_size,
            "requires_every_seed_positive_and_ci_lower_above_zero": True,
            "passes": primary_passes,
            "status": stability_comparison["status"],
        }
    rate_points = [
        {
            "constellation_size": constellation_size,
            "coordinate_bits": config.coordinate_bits,
            "latent_dim": latent_dim,
            "feature_bits": config.feature_bits,
            "stream_bytes": expected_stream_bytes(
                constellation_size, config.coordinate_bits
            ),
            "normalization_bytes": NORMALIZATION.size,
            "total_stream_bytes": expected_stream_bytes(
                constellation_size,
                config.coordinate_bits,
                normalization=True,
            ),
        }
        for latent_dim, constellation_size in zip(
            config.latent_dims, config.matched_constellation_sizes, strict=True
        )
    ]
    result = {
        "experiment": (
            "023_feature_codec_equal_protocol"
            if config.ema_decay is not None
            else "018_feature_codec_multiseed"
        ),
        "config": asdict(config),
        "protocol": {
            "name": (
                "pointconstellation-modelnet40-equal-protocol-feature-codec-v1"
                if config.ema_decay is not None
                else "pointconstellation-modelnet40-matched-feature-codec-v1"
            ),
            "dataset": data_protocol["dataset"],
            "manifest_sha256": data_protocol["manifest_sha256"],
            "data_seed": config.data_seed,
            "input_points": config.num_points,
            "rate_definition": "total serialized stream bits / input points",
            "payload_rate_definition": (
                "byte-aligned feature payload bits / input points"
            ),
            "rate_match": (
                "exact bytes including each format's header and normalization"
            ),
            "rate_points": rate_points,
            "feature_latent_is_not_coordinate_only": True,
            "normalization_payload_bytes_per_object": 8,
            "shared_model_cost_excluded_from_per_cloud_rate": True,
            "ema_checkpoint_selection_uses_calibration_source_only": (
                config.ema_decay is not None
            ),
            "shared_model_cost_reported_as_amortized_bpp": True,
        },
        "data_protocol": data_protocol,
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "per_seed": seed_results,
        "model_independence": {"state_hashes": hashes, "all_unique": True},
        "matched_rate_comparisons": comparisons,
        "stability_reference": stability_comparison,
        "primary_gate": primary_gate,
        "contract_checks": {
            "all_stream_sizes_match_declared_rates": all(
                row["stream_bytes"]
                == expected_stream_bytes(
                    row["constellation_size"],
                    config.coordinate_bits,
                    normalization=True,
                )
                for seed_result in seed_results
                for row in seed_result["per_cloud"]
            ),
            "all_selected_state_hashes_match_models": all(
                seed_result["model"]["selection"]["selected_state_hash"]
                == seed_result["model"]["state_hash"]
                for seed_result in seed_results
            ),
            "all_ema_paths_selected_by_calibration": (
                config.ema_decay is None
                or all(
                    seed_result["model"]["selection"]["kind"] == "ema"
                    and seed_result["model"]["calibration_candidates"]
                    for seed_result in seed_results
                )
            ),
        },
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "elapsed_seconds": max(
            time.perf_counter() - started,
            measured_component_seconds,
            previous_elapsed_seconds or 0.0,
        ),
        "aggregation_elapsed_seconds": time.perf_counter() - started,
    }
    multiseed_metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cuda", "cpu")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = FeatureCodecBenchmarkConfig.from_json(args.config)
    result = run_feature_codec_benchmark(
        config, device_name=args.device, resume=args.resume
    )
    print(
        json.dumps(
            {
                "primary_gate": result["primary_gate"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str(Path(config.output_dir) / "multiseed_metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
