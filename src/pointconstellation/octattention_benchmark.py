"""Experiment 026: pinned OctAttention low-rate benchmark.

The third-party implementation is always executed in its own environment.  This
module owns dataset provenance, actual-byte accounting, official metrics, and
the distinction between released-checkpoint transfer and exact-split retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.codecs import (
    ExternalCodecSpec,
    run_external_codec_batch,
    run_pc_error,
)
from pointconstellation.codecs.gpcc import write_ascii_ply
from pointconstellation.data import file_sha256
from pointconstellation.metrics import chamfer_rmse


def _array_sha256(value: ArrayLike) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


@dataclass(frozen=True)
class OctAttentionArm:
    """One released or exact-split checkpoint family."""

    name: str
    label: str
    checkpoint_subpath: str
    seeds: tuple[int, ...]
    training_content: str

    def __post_init__(self) -> None:
        if self.name not in {"pretrained_transfer", "experiment_019_retrained"}:
            raise ValueError("unknown OctAttention evaluation arm")
        if self.label not in {
            "pretrained_transfer_mpeg_8i_mvub",
            "retrained_exact_experiment_019_train",
        }:
            raise ValueError("OctAttention arm must carry an explicit training label")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("OctAttention arm seeds must be nonempty and unique")
        if self.name == "pretrained_transfer" and len(self.seeds) != 1:
            raise ValueError("the released checkpoint has exactly one identity")
        if self.name == "experiment_019_retrained" and len(self.seeds) < 3:
            raise ValueError("the retrained arm requires at least three seeds")
        if "{seed}" not in self.checkpoint_subpath and len(self.seeds) > 1:
            raise ValueError("multi-seed checkpoint paths must contain {seed}")
        if not self.training_content:
            raise ValueError("training_content cannot be empty")


@dataclass(frozen=True)
class OctAttentionConfig:
    """Pinned code, data, rate, and evaluation protocol for Experiment 026."""

    name: str
    upstream_url: str
    upstream_branch: str
    upstream_commit: str
    license: str
    paper: str
    paper_url: str
    workspace_root: str
    upstream_subpath: str
    environment_python_subpath: str
    environment_manifest_subpath: str
    patch_path: str
    patch_sha256: str
    checkout_diff_sha256: str
    stability_config: str
    expected_stability_manifest_sha256: str
    training_source_subpath: str
    training_manifest_subpath: str
    expected_training_meshes: int
    source_points: int
    position_bits: int
    depths: tuple[int, ...]
    arms: tuple[OctAttentionArm, ...]
    retrain_max_steps: int
    retrain_batch_size: int
    retrain_bptt: int
    retrain_learning_rate: float
    pc_error_executable: str
    splits: tuple[str, ...]
    rate_corridor_bytes: tuple[int, int]
    hypothesis: str
    primary_metric: str
    decision_rule: str
    protocol_note: str
    timeout_seconds: float
    output_dir: str
    max_clouds_per_split: int | None = None

    def __post_init__(self) -> None:
        if self.name != "octattention":
            raise ValueError("this runner supports only OctAttention")
        if not self.upstream_url.startswith("https://"):
            raise ValueError("upstream_url must use HTTPS")
        if self.upstream_branch != "obj":
            raise ValueError("Experiment 026 requires the upstream object branch")
        if len(self.upstream_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.upstream_commit
        ):
            raise ValueError("upstream_commit must be a lowercase full SHA")
        if self.license != "Apache-2.0":
            raise ValueError("unexpected OctAttention license declaration")
        for value in (
            self.patch_sha256,
            self.checkout_diff_sha256,
            self.expected_stability_manifest_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("protocol identities must be lowercase SHA-256")
        if self.expected_training_meshes != 512:
            raise ValueError("Experiment 026 fixes exactly 512 training meshes")
        if self.source_points != 2048 or self.position_bits != 12:
            raise ValueError(
                "Experiment 026 fixes 2,048 source points on a 12-bit grid"
            )
        if self.depths != (4, 5, 6, 7):
            raise ValueError("Experiment 026 fixes octree depths 4, 5, 6, and 7")
        if len(self.arms) != 2 or {arm.name for arm in self.arms} != {
            "pretrained_transfer",
            "experiment_019_retrained",
        }:
            raise ValueError(
                "both pretrained-transfer and exact-retrained arms are required"
            )
        if (
            self.retrain_max_steps < 1
            or self.retrain_batch_size != 32
            or self.retrain_bptt != 1024
            or self.retrain_learning_rate <= 0
        ):
            raise ValueError(
                "OctAttention retraining must declare the matched upstream budget"
            )
        if not self.splits or set(self.splits) - {"validation", "ood"}:
            raise ValueError("splits must contain validation and/or ood")
        if (
            len(self.rate_corridor_bytes) != 2
            or self.rate_corridor_bytes[0] != 20
            or self.rate_corridor_bytes[1] != 200
        ):
            raise ValueError("the predeclared low-rate corridor is 20--200 bytes")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not all(
            (
                self.hypothesis,
                self.primary_metric,
                self.decision_rule,
                self.protocol_note,
            )
        ):
            raise ValueError("the predeclared protocol fields cannot be empty")
        if self.max_clouds_per_split is not None and self.max_clouds_per_split < 2:
            raise ValueError(
                "at least two clouds per split are needed for diversity checks"
            )

    @classmethod
    def from_json(cls, path: Path) -> OctAttentionConfig:
        values = json.loads(path.read_text())
        values["depths"] = tuple(values["depths"])
        values["splits"] = tuple(values["splits"])
        values["rate_corridor_bytes"] = tuple(values["rate_corridor_bytes"])
        values["arms"] = tuple(
            OctAttentionArm(**{**arm, "seeds": tuple(arm["seeds"])})
            for arm in values["arms"]
        )
        return cls(**values)

    def arm(self, name: str) -> OctAttentionArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ValueError(f"unknown OctAttention arm: {name}")


def _quantized_grid(points: ArrayLike, bits: int) -> NDArray[np.uint16]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("source points must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all() or np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("source points must be finite and lie in [-1, 1]")
    levels = (1 << bits) - 1
    quantized = np.rint((array + 1.0) * 0.5 * levels).astype(np.uint16)
    return np.unique(quantized, axis=0)


def _depth_grid(
    points: ArrayLike, depth: int, *, position_bits: int
) -> NDArray[np.uint16]:
    source = _quantized_grid(points, position_bits).astype(np.float64)
    source_levels = (1 << position_bits) - 1
    depth_levels = (1 << depth) - 1
    return np.unique(
        np.rint(source * depth_levels / source_levels).astype(np.uint16), axis=0
    )


def _depth_reconstruction_grid(
    points: NDArray[np.uint16], depth: int, *, position_bits: int
) -> NDArray[np.uint16]:
    depth_levels = (1 << depth) - 1
    position_levels = (1 << position_bits) - 1
    return np.unique(
        np.rint(points.astype(np.float64) * position_levels / depth_levels).astype(
            np.uint16
        ),
        axis=0,
    )


def _model_bytes(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"OctAttention checkpoint is missing: {path}")
    if path.stat().st_size < 1:
        raise RuntimeError(f"OctAttention checkpoint is empty: {path}")
    return path.stat().st_size


def _validate_retrained_identity(
    config: OctAttentionConfig,
    *,
    workspace: Path,
    checkpoint: Path,
    seed: int,
) -> None:
    training_manifest = workspace / config.training_manifest_subpath
    if not training_manifest.is_file():
        raise FileNotFoundError(
            f"OctAttention training manifest is missing: {training_manifest}"
        )
    manifest_sha256 = file_sha256(training_manifest)
    manifest = json.loads(training_manifest.read_text())
    if (
        manifest.get("experiment") != "026_octattention_training_export"
        or manifest.get("training_meshes") != config.expected_training_meshes
        or manifest.get("stability_manifest_sha256")
        != config.expected_stability_manifest_sha256
        or not manifest.get("contract_checks", {}).get("training_split_only", False)
    ):
        raise RuntimeError(
            "OctAttention training manifest violates the exact-split contract"
        )
    record_path = checkpoint.parent / "training_run.json"
    if not record_path.is_file():
        raise FileNotFoundError(
            f"OctAttention training record is missing: {record_path}"
        )
    record = json.loads(record_path.read_text())
    expected = {
        "seed": seed,
        "upstream_commit": config.upstream_commit,
        "checkout_diff_sha256": config.checkout_diff_sha256,
        "training_manifest_sha256": manifest_sha256,
        "training_meshes": config.expected_training_meshes,
        "depths": list(config.depths),
        "position_bits": config.position_bits,
        "max_steps": config.retrain_max_steps,
        "batch_size": config.retrain_batch_size,
        "bptt": config.retrain_bptt,
        "learning_rate": config.retrain_learning_rate,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("OctAttention checkpoint training identity mismatch")
    if record.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise RuntimeError(
            "OctAttention checkpoint SHA-256 differs from training record"
        )


def octattention_codec_spec(
    config: OctAttentionConfig,
    *,
    arm: OctAttentionArm,
    seed: int,
    depth: int,
    workspace: Path | None = None,
) -> ExternalCodecSpec:
    """Build the auditable subprocess contract for one arm/depth point."""

    if seed not in arm.seeds:
        raise ValueError(f"seed {seed} is not declared for arm {arm.name}")
    if depth not in config.depths:
        raise ValueError(f"depth {depth} is not declared")
    root = workspace or Path(config.workspace_root)
    upstream = root / config.upstream_subpath
    python = root / config.environment_python_subpath
    environment = root / config.environment_manifest_subpath
    checkpoint = root / arm.checkpoint_subpath.format(seed=seed)
    if not python.is_file() or not python.stat().st_mode & 0o111:
        raise FileNotFoundError(f"external environment Python is missing: {python}")
    if arm.name == "experiment_019_retrained":
        _validate_retrained_identity(
            config, workspace=root, checkpoint=checkpoint, seed=seed
        )
    adapter = "{upstream_dir}/pointconstellation_adapter.py"
    compress_command = (
        str(python),
        adapter,
        "encode",
        "--checkpoint",
        "{checkpoint_dir}",
        "--position-bits",
        str(config.position_bits),
        "--depth",
        str(depth),
        "--inputs",
        "{inputs}",
        "--streams",
        "{streams}",
    )
    decompress_command = (
        str(python),
        adapter,
        "decode",
        "--checkpoint",
        "{checkpoint_dir}",
        "--streams",
        "{streams}",
        "--reconstructions",
        "{reconstructions}",
    )
    return ExternalCodecSpec(
        name=f"octattention_{arm.name}_s{seed}_d{depth}",
        upstream_url=config.upstream_url,
        upstream_commit=config.upstream_commit,
        upstream_dir=str(upstream),
        checkpoint_dir=str(checkpoint),
        compress_command=compress_command,
        decompress_command=decompress_command,
        position_bits=config.position_bits,
        timeout_seconds=config.timeout_seconds,
        model_bytes=_model_bytes(checkpoint),
        environment_manifest=str(environment),
        environment_variables=(
            ("PYTHONPATH", ""),
            (
                "CUDA_VISIBLE_DEVICES",
                os.environ.get("POINTCONSTELLATION_CODEC_GPU", "0"),
            ),
        ),
        checkout_diff_sha256=config.checkout_diff_sha256,
    )


def export_octattention_training_sources(
    config: OctAttentionConfig, *, workspace: Path | None = None
) -> dict[str, Any]:
    """Export only the exact 512 Experiment 019 training source samples."""

    from pointconstellation.stability_experiment import (
        StabilityExperimentConfig,
        _data_protocol,
        _datasets,
    )

    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    if (
        stability.num_points != config.source_points
        or stability.coordinate_bits != config.position_bits
        or stability.train_samples != config.expected_training_meshes
    ):
        raise RuntimeError("Experiment 019 source protocol differs from Experiment 026")
    if stability.dataset_manifest is None:
        raise RuntimeError("Experiment 026 requires the manifest-backed mesh dataset")
    stability_manifest = Path(stability.dataset_manifest)
    if file_sha256(stability_manifest) != config.expected_stability_manifest_sha256:
        raise RuntimeError("Experiment 019 dataset manifest SHA-256 differs")
    datasets = _datasets(stability)
    training = datasets["train"]
    if len(training) != config.expected_training_meshes:
        raise RuntimeError(
            "Experiment 019 training partition is not exactly 512 meshes"
        )

    root = workspace or Path(config.workspace_root)
    source_dir = root / config.training_source_subpath
    manifest_path = root / config.training_manifest_subpath
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    identities = set()
    for index in range(len(training)):
        sample = training[index]
        identity = (str(sample["family"]), str(sample["model_id"]))
        if identity in identities:
            raise RuntimeError(
                "Experiment 019 training export contains a duplicate mesh"
            )
        identities.add(identity)
        source = sample["source_points"].numpy()
        quantized = _quantized_grid(source, config.position_bits)
        relative = Path(f"{index:04d}_{identity[0]}_{identity[1]}.ply")
        path = source_dir / relative
        if path.exists():
            raise FileExistsError(
                f"training source already exists; use a fresh export root: {path}"
            )
        write_ascii_ply(path, quantized)
        records.append(
            {
                "index": index,
                "split": "train",
                "family": identity[0],
                "model_id": identity[1],
                "sample_id": int(sample["sample_id"]),
                "source_points": len(source),
                "unique_q12_voxels": len(quantized),
                "relative_path": relative.as_posix(),
                "ply_sha256": file_sha256(path),
                "q12_voxel_sha256": _array_sha256(quantized),
            }
        )
    manifest = {
        "version": 1,
        "experiment": "026_octattention_training_export",
        "stability_config": str(stability_path),
        "stability_config_sha256": file_sha256(stability_path),
        "stability_manifest_sha256": file_sha256(stability_manifest),
        "data_protocol": _data_protocol(stability, datasets),
        "training_meshes": len(records),
        "source_points_per_mesh": config.source_points,
        "position_bits": config.position_bits,
        "depths": list(config.depths),
        "records": records,
        "contract_checks": {
            "training_split_only": all(row["split"] == "train" for row in records),
            "exact_training_mesh_count": len(records)
            == config.expected_training_meshes,
            "unique_mesh_identities": len(identities) == len(records),
            "validation_and_ood_absent": True,
        },
    }
    if not all(manifest["contract_checks"].values()):
        raise RuntimeError("OctAttention training export contract failed")
    if manifest_path.exists():
        raise FileExistsError(
            "training manifest already exists; use a fresh export root: "
            f"{manifest_path}"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "training_meshes": len(records),
    }


def octattention_diversity_contract(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reject constant-stream or constant-reconstruction codec failures.

    The check is evaluated independently for every arm/seed/depth point.  It
    conditions on distinct shallow-grid occupancies, so equal source clouds do
    not manufacture a false failure.
    """

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["arm"], int(row["seed"]), int(row["depth"]))
        groups.setdefault(key, []).append(row)
    checks = []
    for (arm, seed, depth), group in sorted(groups.items()):
        input_hashes = {row["codec_input_sha256"] for row in group}
        stream_hashes = {row["stream_sha256"] for row in group}
        reconstruction_hashes = {row["reconstruction_sha256"] for row in group}
        passed = (
            len(input_hashes) >= 2
            and len(stream_hashes) >= 2
            and len(reconstruction_hashes) >= 2
        )
        checks.append(
            {
                "arm": arm,
                "seed": seed,
                "depth": depth,
                "clouds": len(group),
                "unique_codec_inputs": len(input_hashes),
                "unique_streams": len(stream_hashes),
                "unique_reconstructions": len(reconstruction_hashes),
                "passed": passed,
            }
        )
    return {
        "passed": bool(checks) and all(item["passed"] for item in checks),
        "groups": checks,
    }


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["arm"],
        row["seed"],
        row["depth"],
        row["split"],
        row["family"],
        row["model_id"],
        row["sample_id"],
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("OctAttention artifact contains duplicate per-cloud rows")
    return rows


def _rate_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["arm"], row["seed"], row["depth"], row["split"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (arm, seed, depth, split), group in sorted(groups.items()):
        summaries.append(
            {
                "arm": arm,
                "seed": seed,
                "depth": depth,
                "split": split,
                "clouds": len(group),
                "mean_stream_bytes": float(
                    np.mean([row["stream_bytes"] for row in group])
                ),
                "mean_actual_bpp": float(
                    np.mean([row["actual_stream_bpp"] for row in group])
                ),
                "aggregate_chamfer_rmse": math.sqrt(
                    float(np.mean([row["chamfer_mse"] for row in group]))
                ),
                "official_d1_rmse_grid_units": math.sqrt(
                    float(np.mean([row["d1_mse"] for row in group]))
                ),
                "official_d2_rmse_grid_units": math.sqrt(
                    float(np.mean([row["d2_mse"] for row in group]))
                ),
                "mean_encode_seconds": float(
                    np.mean([row["encode_seconds"] for row in group])
                ),
                "mean_decode_seconds": float(
                    np.mean([row["decode_seconds"] for row in group])
                ),
                "model_bytes": group[0]["model_bytes"],
            }
        )
    return summaries


def run_octattention_benchmark(
    config: OctAttentionConfig, *, arm_names: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Run or resume all declared OctAttention arms and depths."""

    from pointconstellation.stability_experiment import (
        StabilityExperimentConfig,
        _data_protocol,
        _datasets,
    )

    selected_arms = (
        config.arms
        if arm_names is None
        else tuple(config.arm(name) for name in arm_names)
    )
    if not selected_arms or len({arm.name for arm in selected_arms}) != len(
        selected_arms
    ):
        raise ValueError("selected OctAttention arms must be nonempty and unique")
    config_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(config_path)
    if stability.num_points != config.source_points:
        raise RuntimeError("Experiment 019 source point count differs")
    if stability.coordinate_bits != config.position_bits:
        raise RuntimeError("official metric grid must be the declared 12-bit grid")
    if (
        stability.dataset_manifest is None
        or file_sha256(Path(stability.dataset_manifest))
        != config.expected_stability_manifest_sha256
    ):
        raise RuntimeError("Experiment 019 dataset manifest identity mismatch")
    patch = Path(config.patch_path)
    if not patch.is_file() or file_sha256(patch) != config.patch_sha256:
        raise RuntimeError("OctAttention compatibility patch identity mismatch")
    executable = Path(config.pc_error_executable)
    if not executable.is_file() or not executable.stat().st_mode & 0o111:
        raise FileNotFoundError(f"pc_error is missing or not executable: {executable}")
    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    workspace = Path(config.workspace_root)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_scratch = output_dir / "metric_scratch"
    metric_scratch.mkdir(exist_ok=True)
    run_manifest = {
        "experiment": "026_octattention_lowrate",
        "selected_arms": [arm.name for arm in selected_arms],
        "config": json.loads(json.dumps(asdict(config))),
        "stability_config_sha256": file_sha256(config_path),
        "pc_error_sha256": file_sha256(executable),
        "data_protocol": data_protocol,
    }
    run_manifest_path = output_dir / "run_manifest.json"
    if run_manifest_path.exists():
        if json.loads(run_manifest_path.read_text()) != run_manifest:
            raise RuntimeError("existing OctAttention run manifest does not match")
    else:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")

    rows_path = output_dir / "per_cloud.jsonl"
    rows = _load_rows(rows_path)
    completed = {_row_key(row) for row in rows}
    resumed_rows = len(rows)
    started = time.perf_counter()
    model_records = []
    for arm in selected_arms:
        for seed in arm.seeds:
            for depth in config.depths:
                spec = octattention_codec_spec(
                    config,
                    arm=arm,
                    seed=seed,
                    depth=depth,
                    workspace=workspace,
                )
                model_records.append(
                    {
                        "arm": arm.name,
                        "arm_label": arm.label,
                        "seed": seed,
                        "depth": depth,
                        "model_bytes": spec.model_bytes,
                        "checkpoint": spec.checkpoint_dir,
                    }
                )
                pending = []
                for split in config.splits:
                    dataset = datasets[split]
                    count = (
                        len(dataset)
                        if config.max_clouds_per_split is None
                        else min(len(dataset), config.max_clouds_per_split)
                    )
                    for index in range(count):
                        sample = dataset[index]
                        key = (
                            arm.name,
                            seed,
                            depth,
                            split,
                            str(sample["family"]),
                            str(sample["model_id"]),
                            int(sample["sample_id"]),
                        )
                        if key in completed:
                            continue
                        pending.append(
                            {
                                "key": key,
                                "sample": sample,
                                "split": split,
                                "source": sample["source_points"].numpy(),
                                "normals": sample["source_normals"].numpy(),
                                "work_dir": output_dir
                                / "streams"
                                / arm.name
                                / f"seed_{seed}"
                                / f"depth_{depth}"
                                / split
                                / f"{sample['family']}_{sample['model_id']}",
                            }
                        )
                if not pending:
                    continue
                results = run_external_codec_batch(
                    spec,
                    tuple(item["source"] for item in pending),
                    work_dirs=tuple(item["work_dir"] for item in pending),
                )
                log_dir = (
                    output_dir
                    / "streams"
                    / arm.name
                    / f"seed_{seed}"
                    / f"depth_{depth}"
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "compress.log").write_text(results[0].compress_output)
                (log_dir / "decompress.log").write_text(results[0].decompress_output)
                for item, codec_result in zip(pending, results, strict=True):
                    source = item["source"]
                    codec_input = _depth_grid(
                        source, depth, position_bits=config.position_bits
                    )
                    expected_reconstruction = _depth_reconstruction_grid(
                        codec_input, depth, position_bits=config.position_bits
                    )
                    decoded_grid = _quantized_grid(
                        codec_result.reconstruction, config.position_bits
                    )
                    depth_occupancy_roundtrip = np.array_equal(
                        decoded_grid, expected_reconstruction
                    )
                    with tempfile.TemporaryDirectory(
                        prefix=f"octattention-d{depth}-", dir=metric_scratch
                    ) as temporary:
                        official = run_pc_error(
                            executable,
                            source,
                            codec_result.reconstruction,
                            item["normals"],
                            work_dir=Path(temporary),
                            position_bits=config.position_bits,
                        )
                    sample = item["sample"]
                    row = {
                        "codec": config.name,
                        "arm": arm.name,
                        "arm_label": arm.label,
                        "training_content": arm.training_content,
                        "seed": seed,
                        "depth": depth,
                        "split": item["split"],
                        "family": str(sample["family"]),
                        "model_id": str(sample["model_id"]),
                        "sample_id": int(sample["sample_id"]),
                        "source_points": len(source),
                        "codec_input_unique_voxels": len(codec_input),
                        "codec_input_sha256": _array_sha256(codec_input),
                        "expected_reconstruction_sha256": _array_sha256(
                            expected_reconstruction
                        ),
                        "decoded_points": len(codec_result.reconstruction),
                        "depth_occupancy_roundtrip": depth_occupancy_roundtrip,
                        "stream_bytes": codec_result.stream_bytes,
                        "actual_stream_bpp": 8.0
                        * codec_result.stream_bytes
                        / len(source),
                        "stream_sha256": codec_result.stream_sha256,
                        "reconstruction_sha256": codec_result.reconstruction_sha256,
                        "upstream_commit": codec_result.upstream_commit,
                        "checkout_diff_sha256": codec_result.checkout_diff_sha256,
                        "model_bytes": spec.model_bytes,
                        "encode_seconds": codec_result.encode_seconds,
                        "decode_seconds": codec_result.decode_seconds,
                        "official_metric_seconds": official.elapsed_seconds,
                        "metric_position_bits": config.position_bits,
                        "chamfer_mse": chamfer_rmse(codec_result.reconstruction, source)
                        ** 2,
                        **official.metrics,
                    }
                    item["work_dir"].joinpath("row.json").write_text(
                        json.dumps(row, indent=2) + "\n"
                    )
                    with rows_path.open("a") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        handle.flush()
                    rows.append(row)
                    completed.add(item["key"])

    diversity = octattention_diversity_contract(rows)
    result = {
        "experiment": "026_octattention_lowrate",
        "config": json.loads(json.dumps(asdict(config))),
        "resumed_rows": resumed_rows,
        "per_cloud_rows": len(rows),
        "model_records": model_records,
        "rate_summaries": _rate_summaries(rows),
        "diversity_contract": diversity,
        "contract_checks": {
            "actual_streams_nonempty": bool(
                rows and all(row["stream_bytes"] > 0 for row in rows)
            ),
            "stream_hashes_present": bool(
                rows and all(len(row["stream_sha256"]) == 64 for row in rows)
            ),
            "independent_decode_hashes_present": bool(
                rows and all(len(row["reconstruction_sha256"]) == 64 for row in rows)
            ),
            "depths_complete": {row["depth"] for row in rows} == set(config.depths),
            "arms_explicitly_labeled": {row["arm_label"] for row in rows}
            == {arm.label for arm in selected_arms},
            "official_metric_grid_q12": all(
                row["metric_position_bits"] == 12 for row in rows
            ),
            "lossless_depth_occupancy_roundtrip": all(
                row["depth_occupancy_roundtrip"] for row in rows
            ),
            "codec_output_diverse": diversity["passed"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "per_cloud_path": str(rows_path),
    }
    if not all(result["contract_checks"].values()):
        raise RuntimeError("OctAttention benchmark contract failed")
    metrics_path = output_dir / "octattention_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--export-training-sources", action="store_true", help="export train only"
    )
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--pc-error", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-clouds-per-split", type=int)
    parser.add_argument(
        "--arm",
        action="append",
        choices=("pretrained_transfer", "experiment_019_retrained"),
        help="evaluate one arm; repeat to select both (default: both)",
    )
    args = parser.parse_args()
    config = OctAttentionConfig.from_json(args.config)
    if args.workspace_root is not None:
        config = replace(config, workspace_root=str(args.workspace_root))
    if args.pc_error is not None:
        config = replace(config, pc_error_executable=str(args.pc_error))
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))
    if args.max_clouds_per_split is not None:
        config = replace(config, max_clouds_per_split=args.max_clouds_per_split)
    if args.export_training_sources:
        result = export_octattention_training_sources(config)
    else:
        result = run_octattention_benchmark(
            config, arm_names=tuple(args.arm) if args.arm else None
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
