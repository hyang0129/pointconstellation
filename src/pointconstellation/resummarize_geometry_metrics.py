"""Resummarize Experiments 021/022 with continuous-mesh and tail metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from pointconstellation.bitstream import decode_constellation, encode_constellation
from pointconstellation.data import MeshSurfaceDataset, file_sha256
from pointconstellation.losses import chamfer_squared_chunked
from pointconstellation.metrics import point_set_error_metrics
from pointconstellation.official_stability import (
    OfficialStabilityConfig,
    _load_models,
    _method_seed,
)
from pointconstellation.selection_baselines import SELECTION_METHODS
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _datasets,
)
from pointconstellation.surface_metrics import mesh_surface_metrics
from pointconstellation.train import select_device

SELECTION_METHODS_TO_RESUMMARIZE = ("fps", "random_best_of_16", "refiner")
HEADROOM_METHODS_TO_RESUMMARIZE = ("adam_multistart",)


@dataclass(frozen=True)
class GeometryResummarizeConfig:
    """Locations and cost bounds for the Experiment 031 mesh resummary."""

    stability_config: str = "configs/experiment_019_stability_modelnet40.json"
    stability_artifact_dir: str = "artifacts/local/experiment_019_stability_modelnet40"
    selection_config: str = "configs/experiment_021_selection_baselines.json"
    selection_rows: str = (
        "artifacts/local/experiment_021_selection_baselines/official_per_cloud.jsonl"
    )
    headroom_rows: str = (
        "artifacts/local/experiment_022_headroom_modelnet40/headroom_per_cloud.jsonl"
    )
    dataset_root_override: str | None = None
    dataset_manifest_override: str | None = None
    splits: tuple[str, ...] = ("validation", "ood")
    max_clouds_per_split: int | None = None
    point_chunk_size: int = 256
    triangle_chunk_size: int = 256
    normal_neighbors: int = 12
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 20_260_831
    confidence_level: float = 0.95
    output_dir: str = "artifacts/local/experiment_031_geometry_resummary"

    def __post_init__(self) -> None:
        if not self.splits or set(self.splits) - {"validation", "ood"}:
            raise ValueError("splits must contain validation and/or ood")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("splits must be unique")
        if self.max_clouds_per_split is not None and self.max_clouds_per_split < 1:
            raise ValueError("max_clouds_per_split must be positive")
        if min(self.point_chunk_size, self.triangle_chunk_size) < 1:
            raise ValueError("mesh metric chunk sizes must be positive")
        if self.normal_neighbors < 3:
            raise ValueError("normal_neighbors must be at least three")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

    @classmethod
    def from_json(cls, path: Path) -> GeometryResummarizeConfig:
        values = json.loads(path.read_text())
        if "splits" in values:
            values["splits"] = tuple(values["splits"])
        return cls(**values)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"per-cloud artifact is absent: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _cloud_limit(
    rows: list[dict[str, Any]], config: GeometryResummarizeConfig
) -> list[dict[str, Any]]:
    if config.max_clouds_per_split is None:
        return rows
    allowed = {}
    for split in config.splits:
        keys = sorted(
            {
                (row["family"], row["model_id"], row["sample_id"])
                for row in rows
                if row["split"] == split
            }
        )
        allowed[split] = set(keys[: config.max_clouds_per_split])
    return [
        row
        for row in rows
        if (row["family"], row["model_id"], row["sample_id"]) in allowed[row["split"]]
    ]


def _replay_coordinates(
    row: dict[str, Any],
    sample: dict[str, Any],
    *,
    stability: StabilityExperimentConfig,
    selection: OfficialStabilityConfig,
    decoder: torch.nn.Module,
    refiner: torch.nn.Module | None,
    device: torch.device,
) -> tuple[NDArray[np.float32], str]:
    stream_hex = row.get("stream_hex")
    if stream_hex is not None:
        stream = bytes.fromhex(stream_hex)
        if (
            "stream_sha256" in row
            and hashlib.sha256(stream).hexdigest() != row["stream_sha256"]
        ):
            raise RuntimeError("stored Experiment 022 stream hash differs")
        packet = decode_constellation(stream)
        if packet.stream_bytes != row["stream_bytes"]:
            raise RuntimeError("stored stream byte count differs")
        return packet.coordinates.astype(np.float32), packet.mode

    source = sample["source_points"].unsqueeze(0).to(device)
    method = row["method"]
    if method == "refiner":
        if refiner is None:
            raise RuntimeError("refiner replay is missing its assigned checkpoint")
        coordinates = refiner(
            source,
            stability.constellation_size,
            decoder=decoder,
            target=source,
            num_output_points=stability.num_points,
        )[0]
        mode = "free"
    else:
        if method not in SELECTION_METHODS:
            raise ValueError(f"cannot replay unknown selection method: {method}")

        def scorer(candidate: torch.Tensor) -> float:
            with torch.no_grad():
                reconstruction = decoder(
                    candidate.unsqueeze(0), num_output_points=stability.num_points
                )
                loss = chamfer_squared_chunked(
                    reconstruction,
                    source,
                    chunk_size=stability.distance_chunk_size,
                )
            return float(loss.item())

        seed = _method_seed(
            selection,
            method=method,
            split=row["split"],
            sample=sample,
        )
        coordinates = SELECTION_METHODS[method](
            source[0],
            stability.constellation_size,
            stability.coordinate_bits,
            seed,
            scorer,
        )
        mode = "fps" if method == "fps" else "strict_subset"
    stream = encode_constellation(
        coordinates.detach().cpu().numpy(),
        bits=stability.coordinate_bits,
        mode=mode,
        output_points=stability.num_points,
    )
    if len(stream) != row["stream_bytes"]:
        raise RuntimeError("replayed Experiment 021 stream has a different byte count")
    return decode_constellation(stream).coordinates.astype(np.float32), mode


def _resummary_row(
    source_experiment: str,
    row: dict[str, Any],
    sample: dict[str, Any],
    coordinates: NDArray[np.float32],
    mode: str,
    reconstruction: NDArray[np.float32],
    dataset: MeshSurfaceDataset,
    config: GeometryResummarizeConfig,
) -> dict[str, Any]:
    source = sample["source_points"].numpy()
    fresh = sample["fresh_points"].numpy()
    source_metrics = point_set_error_metrics(
        reconstruction, source, chunk_size=config.point_chunk_size
    )
    fresh_metrics = point_set_error_metrics(
        reconstruction, fresh, chunk_size=config.point_chunk_size
    )
    surface = mesh_surface_metrics(
        reconstruction,
        dataset.mesh(int(row["sample_id"])),
        point_chunk_size=config.point_chunk_size,
        triangle_chunk_size=config.triangle_chunk_size,
        normal_neighbors=config.normal_neighbors,
    )
    stream = encode_constellation(
        coordinates,
        bits=int(row["coordinate_bits"]),
        mode=mode,
        output_points=len(source),
    )
    return {
        "source_experiment": source_experiment,
        "split": row["split"],
        "method": row["method"],
        "decoder_seed": row["decoder_seed"],
        "refiner_seed": row.get("refiner_seed"),
        "budget": row.get("budget"),
        "family": row["family"],
        "model_id": row["model_id"],
        "sample_id": row["sample_id"],
        "constellation_size": row["constellation_size"],
        "coordinate_bits": row["coordinate_bits"],
        "stream_bytes": len(stream),
        "stream_sha256": hashlib.sha256(stream).hexdigest(),
        "source_p90_euclidean": source_metrics["p90_euclidean"],
        "source_p99_euclidean": source_metrics["p99_euclidean"],
        "source_hausdorff": source_metrics["hausdorff"],
        "fresh_p90_euclidean": fresh_metrics["p90_euclidean"],
        "fresh_p99_euclidean": fresh_metrics["p99_euclidean"],
        "fresh_hausdorff": fresh_metrics["hausdorff"],
        "surface_mse": surface["surface_rmse"] ** 2,
        **surface,
    }


def _arm(row: dict[str, Any]) -> str:
    if row["method"] == "adam_multistart":
        return f"adam_multistart:budget_{row['budget']}"
    return str(row["method"])


def _cloud_draw(
    categories: NDArray[np.str_], rng: np.random.Generator
) -> NDArray[np.int64]:
    unique = np.unique(categories)
    category_draw = unique[rng.integers(0, len(unique), size=len(unique))]
    result = []
    for category in category_draw:
        candidates = np.flatnonzero(categories == category)
        result.extend(
            candidates[rng.integers(0, len(candidates), size=len(candidates))].tolist()
        )
    return np.asarray(result, dtype=np.int64)


def _bootstrap_surface_comparison(
    baseline: NDArray[np.float64],
    candidate: NDArray[np.float64],
    categories: NDArray[np.str_],
    *,
    metric: str,
    config: GeometryResummarizeConfig,
    seed: int,
) -> dict[str, Any]:
    def effect(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
        if metric == "normal_consistency":
            return float(second.mean() - first.mean())
        if metric == "surface_mse":
            first_value = math.sqrt(float(first.mean()))
            second_value = math.sqrt(float(second.mean()))
        else:
            first_value = float(first.mean())
            second_value = float(second.mean())
        return 100.0 * (first_value - second_value) / max(first_value, 1e-12)

    rng = np.random.default_rng(seed)
    draws = np.empty(config.bootstrap_samples, dtype=np.float64)
    for index in range(config.bootstrap_samples):
        decoder_draw = rng.integers(0, baseline.shape[0], size=baseline.shape[0])
        replicate_draw = rng.integers(0, candidate.shape[1], size=candidate.shape[1])
        cloud_draw = _cloud_draw(categories, rng)
        draws[index] = effect(
            baseline[decoder_draw[:, None], cloud_draw[None, :]],
            candidate[
                decoder_draw[:, None, None],
                replicate_draw[None, :, None],
                cloud_draw[None, None, :],
            ],
        )
    alpha = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(draws, (alpha, 1.0 - alpha))
    return {
        "effect_definition": (
            "candidate-minus-FPS mean consistency"
            if metric == "normal_consistency"
            else (
                "relative aggregate surface RMSE improvement over FPS"
                if metric == "surface_mse"
                else "relative mean tail-error improvement over FPS"
            )
        ),
        "effect": effect(baseline, candidate),
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "decoder_count": baseline.shape[0],
        "candidate_replicates": candidate.shape[1],
        "cloud_count": baseline.shape[1],
    }


def summarize_geometry_rows(
    rows: list[dict[str, Any]],
    config: GeometryResummarizeConfig,
    *,
    decoder_seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Compare complete refiner/random/Adam cells with matched FPS geometry."""

    comparisons = []
    aggregate = []
    arms = sorted({_arm(row) for row in rows})
    for split_index, split in enumerate(config.splits):
        split_rows = [row for row in rows if row["split"] == split]
        for arm in arms:
            selected = [row for row in split_rows if _arm(row) == arm]
            if selected:
                aggregate.append(
                    {
                        "split": split,
                        "arm": arm,
                        "rows": len(selected),
                        "surface_rmse": math.sqrt(
                            float(np.mean([row["surface_mse"] for row in selected]))
                        ),
                        "normal_consistency": float(
                            np.mean([row["normal_consistency"] for row in selected])
                        ),
                        "fresh_p90_euclidean": float(
                            np.mean([row["fresh_p90_euclidean"] for row in selected])
                        ),
                        "fresh_p99_euclidean": float(
                            np.mean([row["fresh_p99_euclidean"] for row in selected])
                        ),
                        "fresh_hausdorff": float(
                            np.mean([row["fresh_hausdorff"] for row in selected])
                        ),
                    }
                )
        for candidate_index, candidate_arm in enumerate(
            arm for arm in arms if arm != "fps"
        ):
            candidate_rows = [row for row in split_rows if _arm(row) == candidate_arm]
            fps_rows = [row for row in split_rows if _arm(row) == "fps"]
            candidate_replicates = sorted(
                {
                    row["refiner_seed"]
                    for row in candidate_rows
                    if row["refiner_seed"] is not None
                }
            ) or [None]
            candidate_indexed = {
                (
                    row["decoder_seed"],
                    row["refiner_seed"],
                    row["family"],
                    row["model_id"],
                    row["sample_id"],
                ): row
                for row in candidate_rows
            }
            fps_indexed = {
                (
                    row["decoder_seed"],
                    row["family"],
                    row["model_id"],
                    row["sample_id"],
                ): row
                for row in fps_rows
            }
            cloud_keys = sorted(
                {
                    (row["family"], row["model_id"], row["sample_id"])
                    for row in candidate_rows
                    if all(
                        (decoder_seed, row["family"], row["model_id"], row["sample_id"])
                        in fps_indexed
                        and all(
                            (
                                decoder_seed,
                                replicate,
                                row["family"],
                                row["model_id"],
                                row["sample_id"],
                            )
                            in candidate_indexed
                            for replicate in candidate_replicates
                        )
                        for decoder_seed in decoder_seeds
                    )
                }
            )
            if not cloud_keys:
                comparisons.append(
                    {
                        "split": split,
                        "candidate_arm": candidate_arm,
                        "status": "no_complete_matched_clouds",
                    }
                )
                continue
            categories = np.asarray([cloud[0] for cloud in cloud_keys])
            for metric_index, metric in enumerate(
                (
                    "surface_mse",
                    "normal_consistency",
                    "fresh_p90_euclidean",
                    "fresh_p99_euclidean",
                    "fresh_hausdorff",
                )
            ):
                fps = np.asarray(
                    [
                        [fps_indexed[(seed, *cloud)][metric] for cloud in cloud_keys]
                        for seed in decoder_seeds
                    ]
                )
                candidate = np.asarray(
                    [
                        [
                            [
                                candidate_indexed[(seed, replicate, *cloud)][metric]
                                for cloud in cloud_keys
                            ]
                            for replicate in candidate_replicates
                        ]
                        for seed in decoder_seeds
                    ]
                )
                comparisons.append(
                    {
                        "split": split,
                        "baseline_arm": "fps",
                        "candidate_arm": candidate_arm,
                        "metric": metric,
                        "status": "complete",
                        **_bootstrap_surface_comparison(
                            fps,
                            candidate,
                            categories,
                            metric=metric,
                            config=config,
                            seed=(
                                config.bootstrap_seed
                                + split_index * 100_000
                                + candidate_index * 1_000
                                + metric_index
                            ),
                        ),
                    }
                )
    return {
        "aggregate": aggregate,
        "comparisons": comparisons,
        "comparison_definition": (
            "paired hierarchical category/cloud bootstrap with paired decoder "
            "draws and a resampled refiner replicate factor where applicable"
        ),
    }


def run_geometry_resummary(
    config: GeometryResummarizeConfig,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Replay or decode selected 021/022 messages and write a new artifact."""

    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    if stability.dataset_kind != "mesh_manifest":
        raise ValueError("geometry resummary requires a mesh manifest dataset")
    if config.dataset_root_override is not None:
        stability = replace(stability, dataset_root=config.dataset_root_override)
    if config.dataset_manifest_override is not None:
        stability = replace(
            stability, dataset_manifest=config.dataset_manifest_override
        )
    selection_path = Path(config.selection_config)
    selection = OfficialStabilityConfig.from_json(selection_path)
    selection = replace(
        selection,
        stability_config=config.stability_config,
        stability_artifact_dir=config.stability_artifact_dir,
    )
    datasets = _datasets(stability)
    if any(
        not isinstance(datasets[split], MeshSurfaceDataset) for split in config.splits
    ):
        raise RuntimeError("evaluation splits did not resolve to mesh datasets")

    selection_path_rows = Path(config.selection_rows)
    headroom_path_rows = Path(config.headroom_rows)
    source_rows = [
        ("021_selection_baselines", row)
        for row in _read_jsonl(selection_path_rows)
        if row["split"] in config.splits
        and row["method"] in SELECTION_METHODS_TO_RESUMMARIZE
    ] + [
        ("022_headroom", row)
        for row in _read_jsonl(headroom_path_rows)
        if row["split"] in config.splits
        and row["method"] in HEADROOM_METHODS_TO_RESUMMARIZE
    ]
    limited = _cloud_limit([row for _, row in source_rows], config)
    allowed_ids = {id(row) for row in limited}
    source_rows = [(name, row) for name, row in source_rows if id(row) in allowed_ids]
    device = select_device(device_name)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    loaded: dict[
        tuple[int, int | None], tuple[torch.nn.Module, torch.nn.Module | None]
    ] = {}
    for source_experiment, row in source_rows:
        split = row["split"]
        dataset = datasets[split]
        assert isinstance(dataset, MeshSurfaceDataset)
        sample = dataset[int(row["sample_id"])]
        if (
            str(sample["family"]) != row["family"]
            or str(sample["model_id"]) != row["model_id"]
        ):
            raise RuntimeError("artifact cloud identity differs from current manifest")
        refiner_seed = row.get("refiner_seed") if row["method"] == "refiner" else None
        model_key = (int(row["decoder_seed"]), refiner_seed)
        if model_key not in loaded:
            decoder, refiner, _ = _load_models(
                stability,
                selection,
                decoder_seed=model_key[0],
                refiner_seed=model_key[1],
                device=device,
            )
            loaded[model_key] = decoder, refiner
        decoder, refiner = loaded[model_key]
        with torch.no_grad():
            coordinates, mode = _replay_coordinates(
                row,
                sample,
                stability=stability,
                selection=selection,
                decoder=decoder,
                refiner=refiner,
                device=device,
            )
            reconstruction = decoder(
                torch.from_numpy(coordinates).unsqueeze(0).to(device),
                num_output_points=stability.num_points,
            )[0]
        rows.append(
            _resummary_row(
                source_experiment,
                row,
                sample,
                coordinates,
                mode,
                reconstruction.detach().cpu().numpy(),
                dataset,
                config,
            )
        )

    rows_path = output_dir / "geometry_per_cloud.jsonl"
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_geometry_rows(
        rows, config, decoder_seeds=tuple(stability.decoder_seeds)
    )
    result = {
        "experiment": "031_modelnet40_geometry_resummary",
        "config": json.loads(json.dumps(asdict(config))),
        "source_artifacts": {
            "stability_config_sha256": file_sha256(stability_path),
            "selection_config_sha256": file_sha256(selection_path),
            "selection_rows_sha256": file_sha256(selection_path_rows),
            "headroom_rows_sha256": file_sha256(headroom_path_rows),
        },
        "device": str(device),
        "per_cloud_rows": len(rows),
        "statistics": summary,
        "contract_checks": {
            "source_artifacts_hashed": True,
            "source_cloud_identity_matches_manifest": True,
            "experiment_022_stream_hashes_checked": True,
            "experiment_021_source_only_methods_replayed": True,
            "continuous_mesh_not_finite_sample_used": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "per_cloud_path": str(rows_path),
    }
    (output_dir / "geometry_resummary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_031_geometry_resummarize.json"),
    )
    parser.add_argument("--device")
    parser.add_argument("--max-clouds-per-split", type=int)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = GeometryResummarizeConfig.from_json(args.config)
    overrides = {}
    if args.max_clouds_per_split is not None:
        overrides["max_clouds_per_split"] = args.max_clouds_per_split
    if args.dataset_root is not None:
        overrides["dataset_root_override"] = str(args.dataset_root)
    if args.dataset_manifest is not None:
        overrides["dataset_manifest_override"] = str(args.dataset_manifest)
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)
    if overrides:
        config = replace(config, **overrides)
    run_geometry_resummary(config, device_name=args.device or "auto")


if __name__ == "__main__":
    main()
