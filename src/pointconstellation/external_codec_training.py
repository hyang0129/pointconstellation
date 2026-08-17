"""Export and train pcc_geo_cnn_v2 on the exact Experiment 019 sources.

The generated dataset is intentionally an external artifact.  It contains only
deterministic source point samples from the declared train and calibration
splits, never validation or category-OOD records.  The upstream model continues
to consume its native 64-cubed occupancy blocks.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from pointconstellation.codecs.gpcc import write_ascii_ply
from pointconstellation.data import file_sha256
from pointconstellation.stability_experiment import (
    StabilityExperimentConfig,
    _data_protocol,
    _datasets,
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _bytes_sha256(contiguous.view(np.uint8).tobytes())


@dataclass(frozen=True)
class ExternalTrainingArm:
    """One declared coordinate-grid and octree operating family."""

    name: str
    position_bits: int
    octree_level: int
    lambdas: tuple[str, ...]
    max_steps: int
    batch_size: int = 32
    model_config: str = "c3p"
    alpha: float = 0.75
    gamma: float = 2.0

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("training-arm name must be nonempty without whitespace")
        if not 6 <= self.position_bits <= 12:
            raise ValueError("training-arm position_bits must be between 6 and 12")
        if self.octree_level < 0:
            raise ValueError("training-arm octree_level must be nonnegative")
        resolution = 1 << self.position_bits
        if resolution // (1 << self.octree_level) != 64:
            raise ValueError("every training arm must produce native 64^3 blocks")
        if len(self.lambdas) < 1 or len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("training-arm lambdas must be nonempty and unique")
        if any(float(value) <= 0 for value in self.lambdas):
            raise ValueError("training-arm lambdas must be positive")
        if self.max_steps < 500 or self.batch_size < 1:
            raise ValueError("training budget or batch size is invalid")
        if self.model_config != "c3p":
            raise ValueError("Experiment 020 fixes the published c3p architecture")


@dataclass(frozen=True)
class ExactExternalRetrainConfig:
    """Exact-split external-codec retraining configuration."""

    stability_config: str
    upstream_url: str
    upstream_commit: str
    workspace_root: str
    upstream_subpath: str
    environment_python_subpath: str
    environment_manifest_subpath: str
    training_root_subpath: str
    expected_stability_manifest_sha256: str
    arms: tuple[ExternalTrainingArm, ...]
    expected_dataset_archive_sha256: str | None = None
    expected_dataset_manifest_sha256: str | None = None
    checkout_diff_sha256: str | None = None
    timeout_seconds: float = 172800.0

    def __post_init__(self) -> None:
        if not self.upstream_url.startswith("https://"):
            raise ValueError("upstream_url must use HTTPS")
        if len(self.upstream_commit) != 40:
            raise ValueError("upstream_commit must be a full SHA")
        if len(self.expected_stability_manifest_sha256) != 64:
            raise ValueError("stability manifest must be pinned by SHA-256")
        if not self.arms or len({arm.name for arm in self.arms}) != len(self.arms):
            raise ValueError("training arms must be nonempty and uniquely named")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for expected in (
            self.expected_dataset_archive_sha256,
            self.expected_dataset_manifest_sha256,
            self.checkout_diff_sha256,
        ):
            if expected is not None and len(expected) != 64:
                raise ValueError("expected dataset hashes must be SHA-256 values")

    @classmethod
    def from_json(cls, path: Path) -> ExactExternalRetrainConfig:
        values = json.loads(path.read_text())
        values["arms"] = tuple(
            ExternalTrainingArm(
                **{
                    **arm,
                    "lambdas": tuple(arm["lambdas"]),
                }
            )
            for arm in values["arms"]
        )
        return cls(**values)

    def arm(self, name: str) -> ExternalTrainingArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ValueError(f"unknown training arm: {name}")


def _quantize_source(points: np.ndarray, position_bits: int) -> np.ndarray:
    levels = (1 << position_bits) - 1
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("source cloud must have shape N x 3")
    if not np.isfinite(points).all() or points.min() < -1 or points.max() > 1:
        raise ValueError("source cloud must be finite and lie in [-1, 1]")
    return np.unique(np.rint((points + 1.0) * 0.5 * levels).astype(np.int64), axis=0)


def _native_blocks(points: np.ndarray, arm: ExternalTrainingArm) -> list[np.ndarray]:
    block_size = 64
    block_ids = points // block_size
    unique_ids = np.unique(block_ids, axis=0)
    blocks = []
    for block_id in unique_ids:
        mask = np.all(block_ids == block_id, axis=1)
        local = points[mask] - block_id * block_size
        if len(local):
            blocks.append(local.astype(np.float32))
    if not blocks:
        raise RuntimeError(f"quantization produced no blocks for arm {arm.name}")
    return blocks


def _write_dataset_tree(
    config: ExactExternalRetrainConfig, root: Path
) -> dict[str, Any]:
    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    manifest_path = Path(stability.dataset_manifest)
    if file_sha256(manifest_path) != config.expected_stability_manifest_sha256:
        raise RuntimeError("stability mesh manifest SHA-256 differs from declaration")
    datasets = _datasets(stability)
    protocol = _data_protocol(stability, datasets)
    records: list[dict[str, Any]] = []
    split_map = {"train": "train", "calibration": "test"}
    forbidden_ids = {
        str(sample["model_id"])
        for split in ("validation", "ood")
        for sample in datasets[split]
    }
    exported_ids: set[str] = set()
    for source_split, upstream_split in split_map.items():
        dataset = datasets[source_split]
        for sample_index in range(len(dataset)):
            sample = dataset[sample_index]
            model_id = str(sample["model_id"])
            if model_id in forbidden_ids:
                raise RuntimeError("external training export contains a test identity")
            if model_id in exported_ids:
                raise RuntimeError("external training identity is duplicated")
            exported_ids.add(model_id)
            source = sample["source_points"].numpy().astype(np.float32, copy=False)
            arm_records = []
            for arm in config.arms:
                quantized = _quantize_source(source, arm.position_bits)
                blocks = _native_blocks(quantized, arm)
                files = []
                arm_dir = root / "datasets" / arm.name / upstream_split
                arm_dir.mkdir(parents=True, exist_ok=True)
                for block_index, block in enumerate(blocks):
                    relative = (
                        Path("datasets")
                        / arm.name
                        / upstream_split
                        / (f"{model_id}_{block_index:03d}.ply")
                    )
                    path = root / relative
                    write_ascii_ply(path, block)
                    files.append(
                        {
                            "path": relative.as_posix(),
                            "points": len(block),
                            "sha256": file_sha256(path),
                        }
                    )
                arm_records.append(
                    {
                        "arm": arm.name,
                        "position_bits": arm.position_bits,
                        "octree_level": arm.octree_level,
                        "unique_voxels": len(quantized),
                        "blocks": len(blocks),
                        "files": files,
                    }
                )
            records.append(
                {
                    "source_split": source_split,
                    "upstream_split": upstream_split,
                    "category": str(sample["family"]),
                    "model_id": model_id,
                    "sample_id": int(sample["sample_id"]),
                    "source_points": len(source),
                    "source_sha256": _array_sha256(source),
                    "arms": arm_records,
                }
            )
    export_config = json.loads(json.dumps(asdict(config)))
    # Artifact hashes cannot be embedded in the artifact they identify.  Keep
    # the export recipe stable before and after those hashes are pinned.
    export_config["expected_dataset_archive_sha256"] = None
    export_config["expected_dataset_manifest_sha256"] = None
    # Training checkout portability does not affect source quantization.
    export_config["upstream_subpath"] = "upstream"
    export_config.pop("checkout_diff_sha256", None)
    manifest = {
        "version": 1,
        "experiment": "020_exact_external_retrain",
        "stability_config": str(stability_path),
        "stability_config_sha256": file_sha256(stability_path),
        "stability_manifest_sha256": file_sha256(manifest_path),
        "data_protocol": protocol,
        "split_contract": {
            "train_source_split": "train",
            "calibration_source_split": "calibration",
            "upstream_train_directory": "train",
            "upstream_calibration_directory": "test",
            "validation_and_ood_exported": False,
        },
        "config": export_config,
        "records": records,
    }
    manifest_bytes = _json_bytes(manifest)
    (root / "training_dataset_manifest.json").write_bytes(manifest_bytes)
    return {
        "manifest": manifest,
        "manifest_sha256": _bytes_sha256(manifest_bytes),
    }


def _add_tree_to_tar(tar: tarfile.TarFile, root: Path) -> None:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        info = tar.gettarinfo(str(path), arcname=relative)
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        if path.is_file():
            with path.open("rb") as handle:
                tar.addfile(info, handle)
        else:
            tar.addfile(info)


def export_training_archive(
    config: ExactExternalRetrainConfig, output_path: Path
) -> dict[str, Any]:
    """Create a deterministic, exact-source external training dataset."""

    if output_path.exists():
        raise FileExistsError(f"training archive already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pc-external-train-") as temporary:
        root = Path(temporary)
        metadata = _write_dataset_tree(config, root)
        with (
            output_path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
            tarfile.open(fileobj=zipped, mode="w") as tar,
        ):
            _add_tree_to_tar(tar, root)
    metadata.update(
        {
            "archive": str(output_path),
            "archive_bytes": output_path.stat().st_size,
            "archive_sha256": file_sha256(output_path),
        }
    )
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    sidecar.write_bytes(
        _json_bytes(
            {key: value for key, value in metadata.items() if key != "manifest"}
        )
    )
    return metadata


def export_evaluation_archive(
    config: ExactExternalRetrainConfig,
    output_path: Path,
    *,
    max_clouds_per_split: int,
) -> dict[str, Any]:
    """Seal fixed validation/OOD sources separately from all training data."""

    if max_clouds_per_split < 1:
        raise ValueError("max_clouds_per_split must be positive")
    if output_path.exists():
        raise FileExistsError(f"evaluation archive already exists: {output_path}")
    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    manifest_path = Path(stability.dataset_manifest)
    if file_sha256(manifest_path) != config.expected_stability_manifest_sha256:
        raise RuntimeError("stability mesh manifest SHA-256 differs from declaration")
    datasets = _datasets(stability)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pc-external-eval-") as temporary:
        root = Path(temporary)
        records = []
        for split in ("validation", "ood"):
            dataset = datasets[split]
            count = min(len(dataset), max_clouds_per_split)
            for sample_index in range(count):
                sample = dataset[sample_index]
                source = sample["source_points"].numpy().astype(np.float32, copy=False)
                normals = (
                    sample["source_normals"].numpy().astype(np.float32, copy=False)
                )
                identity = f"{sample['family']}_{sample['model_id']}"
                source_relative = Path("arrays") / split / f"{identity}.source.npy"
                normals_relative = Path("arrays") / split / f"{identity}.normals.npy"
                source_path = root / source_relative
                normals_path = root / normals_relative
                source_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(source_path, source, allow_pickle=False)
                np.save(normals_path, normals, allow_pickle=False)
                records.append(
                    {
                        "split": split,
                        "category": str(sample["family"]),
                        "model_id": str(sample["model_id"]),
                        "sample_id": int(sample["sample_id"]),
                        "source_points": len(source),
                        "source_path": source_relative.as_posix(),
                        "source_sha256": file_sha256(source_path),
                        "normals_path": normals_relative.as_posix(),
                        "normals_sha256": file_sha256(normals_path),
                    }
                )
        manifest = {
            "version": 1,
            "experiment": "020_exact_external_evaluation",
            "stability_config_sha256": file_sha256(stability_path),
            "stability_manifest_sha256": file_sha256(manifest_path),
            "data_protocol": _data_protocol(stability, datasets),
            "selection": {
                "policy": "first_manifest_records",
                "max_clouds_per_split": max_clouds_per_split,
                "splits": ["validation", "ood"],
                "created_after_training_launch": True,
                "available_to_training_processes": False,
            },
            "records": records,
        }
        manifest_bytes = _json_bytes(manifest)
        (root / "evaluation_manifest.json").write_bytes(manifest_bytes)
        with (
            output_path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
            tarfile.open(fileobj=zipped, mode="w") as tar,
        ):
            _add_tree_to_tar(tar, root)
    result = {
        "archive": str(output_path),
        "archive_bytes": output_path.stat().st_size,
        "archive_sha256": file_sha256(output_path),
        "manifest_sha256": _bytes_sha256(manifest_bytes),
        "records": len(records),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_bytes(
        _json_bytes(result)
    )
    return result


def _safe_extract(archive: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"dataset destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe training archive member: {member.name}")
        tar.extractall(destination, members=members)


def extract_training_archive(
    config: ExactExternalRetrainConfig, archive: Path, destination: Path
) -> dict[str, Any]:
    """Verify and extract a generated dataset on the external host."""

    archive_sha = file_sha256(archive)
    if (
        config.expected_dataset_archive_sha256 is not None
        and archive_sha != config.expected_dataset_archive_sha256
    ):
        raise RuntimeError("external training archive SHA-256 differs from declaration")
    _safe_extract(archive, destination)
    manifest_path = destination / "training_dataset_manifest.json"
    manifest_sha = file_sha256(manifest_path)
    if (
        config.expected_dataset_manifest_sha256 is not None
        and manifest_sha != config.expected_dataset_manifest_sha256
    ):
        raise RuntimeError(
            "external training manifest SHA-256 differs from declaration"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest["stability_manifest_sha256"] != (
        config.expected_stability_manifest_sha256
    ):
        raise RuntimeError("extracted dataset references a different stability split")
    if manifest["split_contract"]["validation_and_ood_exported"] is not False:
        raise RuntimeError("external training dataset contains forbidden test records")
    return {
        "archive_sha256": archive_sha,
        "manifest_sha256": manifest_sha,
        "records": len(manifest["records"]),
    }


def _git_output(upstream: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(upstream), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_diff_sha256(upstream: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(upstream), "diff", "--binary", "HEAD"),
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _checkpoint_files(checkpoint: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(checkpoint).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file() and path.name != "training_run.json"
    ]


def run_training_point(
    config: ExactExternalRetrainConfig,
    *,
    arm_name: str,
    rate_lambda: str,
    gpu: str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Train or resume one declared external lambda point."""

    arm = config.arm(arm_name)
    if rate_lambda not in arm.lambdas:
        raise ValueError(f"lambda {rate_lambda} is not declared for arm {arm.name}")
    workspace = Path(config.workspace_root)
    upstream = workspace / config.upstream_subpath
    python = workspace / config.environment_python_subpath
    environment_path = workspace / config.environment_manifest_subpath
    training_root = workspace / config.training_root_subpath
    dataset = training_root / "dataset" / "datasets" / arm.name
    if not python.is_file() or not environment_path.is_file() or not dataset.is_dir():
        raise FileNotFoundError(
            "external environment or exact training dataset is missing"
        )
    commit = _git_output(upstream, "rev-parse", "HEAD")
    if commit != config.upstream_commit:
        raise RuntimeError("external upstream commit differs from declaration")
    actual_diff = _git_diff_sha256(upstream)
    expected_diff = config.checkout_diff_sha256 or hashlib.sha256(b"").hexdigest()
    if actual_diff != expected_diff:
        raise RuntimeError("external training checkout patch identity mismatch")
    dataset_manifest = training_root / "dataset" / "training_dataset_manifest.json"
    if (
        config.expected_dataset_manifest_sha256 is not None
        and file_sha256(dataset_manifest) != config.expected_dataset_manifest_sha256
    ):
        raise RuntimeError("training dataset manifest differs from declaration")
    budget = arm.max_steps if max_steps is None else max_steps
    if budget < 500 or budget > arm.max_steps:
        raise ValueError(
            "max_steps override must lie between 500 and the declared budget"
        )
    checkpoint = training_root / "checkpoints" / arm.name / rate_lambda
    checkpoint.mkdir(parents=True, exist_ok=True)
    run_path = checkpoint / "training_run.json"
    if run_path.exists():
        existing = json.loads(run_path.read_text())
        if int(existing["max_steps"]) >= budget and (checkpoint / "done").exists():
            return existing
    command = (
        str(python),
        str(upstream / "src" / "tr_train.py"),
        str(dataset / "**" / "*.ply"),
        str(checkpoint),
        "--resolution",
        "64",
        "--lmbda",
        rate_lambda,
        "--alpha",
        str(arm.alpha),
        "--gamma",
        str(arm.gamma),
        "--batch_size",
        str(arm.batch_size),
        "--model_config",
        arm.model_config,
        "--max_steps",
        str(budget),
    )
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "CUDA_CACHE_MAXSIZE": str(2 * 1024**3),
            "TF_CUDNN_USE_AUTOTUNE": "0",
            "LD_LIBRARY_PATH": str(workspace / "env" / "lib")
            + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""),
        }
    )
    log_path = checkpoint / "training.log"
    started_at = time.time()
    with log_path.open("a") as log:
        completed = subprocess.run(
            command,
            cwd=upstream / "src",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=config.timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        message = f"external training failed with status {completed.returncode}"
        raise RuntimeError(f"{message}; see {log_path}")
    if not (checkpoint / "done").is_file():
        raise RuntimeError("upstream training returned without its done marker")
    files = _checkpoint_files(checkpoint)
    if not any(file["path"].startswith("model.ckpt-") for file in files):
        raise RuntimeError("external training produced no model checkpoint")
    checkpoint_state = (checkpoint / "checkpoint").read_text().splitlines()[0]
    checkpoint_prefix = checkpoint_state.split('"', 2)[1]
    checkpoint_step = int(checkpoint_prefix.rsplit("-", 1)[1])
    result = {
        "version": 1,
        "experiment": "020_exact_external_retrain",
        "arm": arm.name,
        "lambda": rate_lambda,
        "max_steps": budget,
        "checkpoint_step": checkpoint_step,
        "stopped_before_max_steps": checkpoint_step < budget,
        "batch_size": arm.batch_size,
        "model_config": arm.model_config,
        "upstream_url": config.upstream_url,
        "upstream_commit": commit,
        "checkout_diff_sha256": actual_diff,
        "environment": json.loads(environment_path.read_text()),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "command": list(command),
        "gpu": gpu,
        "elapsed_seconds": time.time() - started_at,
        "checkpoint_bytes": sum(file["bytes"] for file in files),
        "checkpoint_files": files,
    }
    run_path.write_bytes(_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output", type=Path, required=True)
    evaluation_parser = subparsers.add_parser("export-evaluation")
    evaluation_parser.add_argument("--output", type=Path, required=True)
    evaluation_parser.add_argument("--max-clouds-per-split", type=int, default=16)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--archive", type=Path, required=True)
    extract_parser.add_argument("--destination", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--arm", required=True)
    train_parser.add_argument("--lambda", dest="rate_lambda", required=True)
    train_parser.add_argument("--gpu", default="0")
    train_parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = ExactExternalRetrainConfig.from_json(args.config)
    if args.command == "export":
        result = export_training_archive(config, args.output)
        result.pop("manifest")
    elif args.command == "export-evaluation":
        result = export_evaluation_archive(
            config,
            args.output,
            max_clouds_per_split=args.max_clouds_per_split,
        )
    elif args.command == "extract":
        result = extract_training_archive(config, args.archive, args.destination)
    else:
        result = run_training_point(
            config,
            arm_name=args.arm,
            rate_lambda=args.rate_lambda,
            gpu=args.gpu,
            max_steps=args.max_steps,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
