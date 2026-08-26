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
from torch.utils.data import DataLoader

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
    validation_samples: int = 128
    category_ood_samples: int = 32
    batch_size: int = 4
    epochs: int = 2
    learning_rate: float = 1e-3
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


def _datasets(config: FeatureCodecBenchmarkConfig) -> dict[str, MeshSurfaceDataset]:
    datasets = {
        split: MeshSurfaceDataset(
            Path(config.dataset_root),
            Path(config.dataset_manifest),
            split=split,
            num_points=config.num_points,
            seed=config.data_seed,
            verify_hashes=True,
            training_target="source",
        )
        for split in ("train", "validation", "category_ood")
    }
    expected = {
        "train": config.train_samples,
        "validation": config.validation_samples,
        "category_ood": config.category_ood_samples,
    }
    for split, count in expected.items():
        if len(datasets[split]) != count:
            raise ValueError(
                f"manifest split {split} has {len(datasets[split])} records; "
                f"config expects {count}"
            )
    return datasets


def _parameter_count(codec: VariableFeatureCodec) -> int:
    return sum(parameter.numel() for parameter in codec.parameters())


def _train_seed(
    config: FeatureCodecBenchmarkConfig,
    *,
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
    datasets = _datasets(config)
    loader = DataLoader(
        datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(model_seed),
    )
    started = time.perf_counter()
    history = []
    update_count = 0
    for epoch in range(config.epochs):
        total = 0.0
        examples = 0
        codec.train()
        for batch_index, batch in enumerate(loader):
            latent_dim = config.latent_dims[
                (epoch * len(loader) + batch_index) % len(config.latent_dims)
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
            total += float(loss.item()) * len(points)
            examples += len(points)
            update_count += 1
        history.append(
            {
                "epoch": epoch + 1,
                "training_chamfer_rmse": math.sqrt(total / examples),
            }
        )
        print(
            json.dumps(
                {
                    "stage": "feature_codec",
                    "model_seed": model_seed,
                    **history[-1],
                }
            ),
            flush=True,
        )
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
    return codec.eval(), {
        "history": history,
        "optimizer_updates": update_count,
        "training_elapsed_seconds": time.perf_counter() - started,
        "parameters": _parameter_count(codec),
        "state_hash": _state_hash(codec),
        "encoder_checkpoint_bytes": (output_dir / "encoder.pt").stat().st_size,
        "decoder_checkpoint_bytes": (output_dir / "decoder.pt").stat().st_size,
        "decoder_state_dicts": decoder_files,
    }


def _evaluate_seed(
    config: FeatureCodecBenchmarkConfig,
    *,
    codec: VariableFeatureCodec,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], float]:
    datasets = _datasets(config)
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
    metrics_path = output_dir / "multiseed_metrics.json"
    previous_elapsed_seconds = None
    if resume and metrics_path.exists():
        previous_elapsed_seconds = json.loads(metrics_path.read_text()).get(
            "elapsed_seconds"
        )
    manifest = load_mesh_manifest(Path(config.dataset_manifest))
    started = time.perf_counter()
    seed_results = []
    for model_seed in config.model_seeds:
        seed_dir = output_dir / f"seed_{model_seed}"
        metrics_path = seed_dir / "benchmark_metrics.json"
        if resume and metrics_path.exists():
            seed_result = json.loads(metrics_path.read_text())
            if (
                seed_result.get("model_seed") != model_seed
                or seed_result.get("data_seed") != config.data_seed
            ):
                raise RuntimeError("existing feature-codec seed does not match config")
        else:
            codec, model = _train_seed(
                config,
                model_seed=model_seed,
                device=device,
                output_dir=seed_dir / "model",
            )
            rows, evaluation_seconds = _evaluate_seed(
                config,
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
            metrics_path.write_text(json.dumps(seed_result, indent=2) + "\n")
            with (seed_dir / "per_cloud.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
        seed_results.append(seed_result)
    hashes = [result["model"]["state_hash"] for result in seed_results]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("independent feature-codec seeds produced duplicate hashes")
    comparisons = _reference_comparisons(config, seed_results)
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
    result = {
        "config": asdict(config),
        "protocol": {
            "name": "pointconstellation-modelnet40-matched-feature-codec-v1",
            "dataset": manifest["dataset"],
            "manifest_sha256": file_sha256(Path(config.dataset_manifest)),
            "data_seed": config.data_seed,
            "input_points": config.num_points,
            "rate_definition": "total serialized stream bits / input points",
            "payload_rate_definition": (
                "byte-aligned feature payload bits / input points"
            ),
            "rate_match": (
                "exact bytes including each format's header and normalization"
            ),
            "feature_latent_is_not_coordinate_only": True,
            "normalization_payload_bytes_per_object": 8,
            "shared_model_cost_excluded_from_per_cloud_rate": True,
            "shared_model_cost_reported_as_amortized_bpp": True,
        },
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "per_seed": seed_results,
        "model_independence": {"state_hashes": hashes, "all_unique": True},
        "matched_rate_comparisons": comparisons,
        "primary_gate": {
            "constellation_size": config.primary_constellation_size,
            "requires_every_seed_positive_and_ci_lower_above_zero": True,
            "passes": primary_passes,
        },
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "elapsed_seconds": max(
            time.perf_counter() - started,
            measured_component_seconds,
            previous_elapsed_seconds or 0.0,
        ),
        "aggregation_elapsed_seconds": time.perf_counter() - started,
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
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
