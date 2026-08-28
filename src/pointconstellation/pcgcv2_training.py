"""Prepare and launch exact-split PCGCv2 retraining for Experiment 027."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pointconstellation.codecs.gpcc import write_ascii_ply
from pointconstellation.codecs.pcgcv2 import (
    PCGCV2_UPSTREAM_COMMIT,
    PCGCV2_UPSTREAM_URL,
)
from pointconstellation.data import file_sha256


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _git(upstream: Path, *arguments: str) -> str:
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


@dataclass(frozen=True)
class Pcgcv2RetrainArm:
    """One exact-split voxel precision and rate-objective setting."""

    name: str
    position_bits: int
    alpha: float
    beta: float
    epochs: int
    batch_size: int = 8
    learning_rate: float = 0.0008

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("PCGCv2 retrain arm name must be nonempty")
        if self.position_bits not in {6, 7, 8}:
            raise ValueError("PCGCv2 retraining precision must be 6, 7, or 8 bits")
        if self.alpha <= 0 or self.beta <= 0 or self.learning_rate <= 0:
            raise ValueError("PCGCv2 retraining objective values must be positive")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("PCGCv2 retraining budget must be positive")


@dataclass(frozen=True)
class Pcgcv2RetrainConfig:
    """Pinned exact-split export and external training paths."""

    stability_config: str
    expected_stability_manifest_sha256: str
    workspace_root: str
    upstream_subpath: str
    environment_python_subpath: str
    environment_manifest_subpath: str
    training_adapter_script: str
    dataset_root_subpath: str
    checkpoint_root_subpath: str
    arms: tuple[Pcgcv2RetrainArm, ...]
    upstream_url: str = PCGCV2_UPSTREAM_URL
    upstream_commit: str = PCGCV2_UPSTREAM_COMMIT
    checkout_diff_sha256: str | None = None
    seed: int = 20260826
    timeout_seconds: float = 172800.0

    def __post_init__(self) -> None:
        if self.upstream_url != PCGCV2_UPSTREAM_URL:
            raise ValueError("PCGCv2 retrain URL differs from the pinned repository")
        if self.upstream_commit != PCGCV2_UPSTREAM_COMMIT:
            raise ValueError("PCGCv2 retrain commit differs from the pinned checkout")
        if len(self.expected_stability_manifest_sha256) != 64:
            raise ValueError("stability manifest must be pinned by SHA-256")
        if (
            self.checkout_diff_sha256 is not None
            and len(self.checkout_diff_sha256) != 64
        ):
            raise ValueError("checkout_diff_sha256 must be a SHA-256")
        if not self.arms or len({arm.name for arm in self.arms}) != len(self.arms):
            raise ValueError("PCGCv2 retrain arms must be nonempty and unique")
        if self.seed < 0 or self.timeout_seconds <= 0:
            raise ValueError("PCGCv2 retrain seed or timeout is invalid")

    @classmethod
    def from_json(cls, path: Path) -> Pcgcv2RetrainConfig:
        values = json.loads(path.read_text())
        values["arms"] = tuple(Pcgcv2RetrainArm(**arm) for arm in values["arms"])
        return cls(**values)

    def arm(self, name: str) -> Pcgcv2RetrainArm:
        try:
            return next(arm for arm in self.arms if arm.name == name)
        except StopIteration as error:
            raise ValueError(f"unknown PCGCv2 retrain arm: {name}") from error


def _quantize(points: np.ndarray, position_bits: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("PCGCv2 training source must have shape (N, 3)")
    if not np.isfinite(points).all() or points.min() < -1 or points.max() > 1:
        raise ValueError("PCGCv2 training source must be finite and in [-1, 1]")
    levels = (1 << position_bits) - 1
    return np.unique(np.rint((points + 1.0) * 0.5 * levels).astype(np.int64), axis=0)


def export_exact_split(config: Pcgcv2RetrainConfig) -> dict[str, Any]:
    """Materialize train/calibration coordinates without test identities."""

    from pointconstellation.stability_experiment import (
        StabilityExperimentConfig,
        _data_protocol,
        _datasets,
    )

    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    manifest_path = Path(stability.dataset_manifest)
    if file_sha256(manifest_path) != config.expected_stability_manifest_sha256:
        raise RuntimeError("stability manifest SHA-256 differs from the declaration")
    datasets = _datasets(stability)
    forbidden = {
        str(datasets[split][index]["model_id"])
        for split in ("validation", "ood")
        for index in range(len(datasets[split]))
    }
    workspace = Path(config.workspace_root)
    dataset_root = workspace / config.dataset_root_subpath
    manifest_output = dataset_root / "pcgcv2_training_manifest.json"
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(f"PCGCv2 dataset root is not empty: {dataset_root}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    records = []
    exported: set[str] = set()
    for split in ("train", "calibration"):
        dataset = datasets[split]
        for index in range(len(dataset)):
            sample = dataset[index]
            model_id = str(sample["model_id"])
            if model_id in forbidden or model_id in exported:
                raise RuntimeError("PCGCv2 training export violates the exact split")
            exported.add(model_id)
            source = sample["source_points"].numpy().astype(np.float32, copy=False)
            arm_records = []
            for arm in config.arms:
                quantized = _quantize(source, arm.position_bits)
                relative = Path(arm.name) / split / f"{model_id}.ply"
                output = dataset_root / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                write_ascii_ply(output, quantized)
                arm_records.append(
                    {
                        "arm": arm.name,
                        "position_bits": arm.position_bits,
                        "path": relative.as_posix(),
                        "unique_voxels": len(quantized),
                        "sha256": file_sha256(output),
                    }
                )
            records.append(
                {
                    "split": split,
                    "family": str(sample["family"]),
                    "model_id": model_id,
                    "sample_id": int(sample["sample_id"]),
                    "source_points": len(source),
                    "arms": arm_records,
                }
            )
    manifest = {
        "version": 1,
        "experiment": "027_pcgcv2_exact_retrain",
        "config": asdict(config),
        "stability_config_sha256": file_sha256(stability_path),
        "stability_manifest_sha256": file_sha256(manifest_path),
        "data_protocol": _data_protocol(stability, datasets),
        "split_contract": {
            "included": ["train", "calibration"],
            "validation_and_ood_exported": False,
        },
        "records": records,
    }
    manifest_output.write_text(_json(manifest))
    return {
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_output),
        "manifest_sha256": file_sha256(manifest_output),
        "records": len(records),
    }


def run_training_arm(
    config: Pcgcv2RetrainConfig, arm_name: str, *, gpu: str
) -> dict[str, Any]:
    """Launch one declared arm in the isolated PCGCv2 environment."""

    arm = config.arm(arm_name)
    workspace = Path(config.workspace_root)
    upstream = workspace / config.upstream_subpath
    python = workspace / config.environment_python_subpath
    environment_path = workspace / config.environment_manifest_subpath
    adapter = Path(config.training_adapter_script)
    dataset_root = workspace / config.dataset_root_subpath
    dataset_manifest = dataset_root / "pcgcv2_training_manifest.json"
    for path, label in (
        (upstream, "upstream checkout"),
        (python, "environment Python"),
        (environment_path, "environment manifest"),
        (adapter, "training adapter"),
        (dataset_manifest, "training manifest"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"PCGCv2 {label} is missing: {path}")
    if _git(upstream, "rev-parse", "HEAD") != config.upstream_commit:
        raise RuntimeError("PCGCv2 training checkout commit differs")
    actual_diff = _git_diff_sha256(upstream)
    expected_diff = config.checkout_diff_sha256 or hashlib.sha256(b"").hexdigest()
    if actual_diff != expected_diff:
        raise RuntimeError("PCGCv2 training checkout patch differs")
    manifest = json.loads(dataset_manifest.read_text())
    if (
        manifest["stability_manifest_sha256"]
        != (config.expected_stability_manifest_sha256)
        or manifest["split_contract"]["validation_and_ood_exported"] is not False
    ):
        raise RuntimeError("PCGCv2 training manifest violates the split contract")

    output = workspace / config.checkpoint_root_subpath / arm.name
    output.mkdir(parents=True, exist_ok=True)
    final_checkpoint = output / f"epoch_{arm.epochs - 1}.pth"
    run_path = output / "training_run.json"
    if run_path.is_file() and final_checkpoint.is_file():
        return json.loads(run_path.read_text())
    command = (
        str(python),
        str(adapter),
        "--upstream-dir",
        str(upstream),
        "--train-dir",
        str(dataset_root / arm.name / "train"),
        "--calibration-dir",
        str(dataset_root / arm.name / "calibration"),
        "--output-dir",
        str(output),
        "--alpha",
        str(arm.alpha),
        "--beta",
        str(arm.beta),
        "--epochs",
        str(arm.epochs),
        "--batch-size",
        str(arm.batch_size),
        "--learning-rate",
        str(arm.learning_rate),
        "--seed",
        str(config.seed),
    )
    started = time.perf_counter()
    with (output / "training.log").open("a") as log:
        completed = subprocess.run(
            command,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=config.timeout_seconds,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONHASHSEED": "0"},
        )
    if completed.returncode:
        raise RuntimeError(
            f"PCGCv2 training failed with status {completed.returncode}; see "
            f"{output / 'training.log'}"
        )
    if not final_checkpoint.is_file():
        raise RuntimeError("PCGCv2 training produced no final deployment checkpoint")
    result = {
        "experiment": "027_pcgcv2_exact_retrain",
        "arm": asdict(arm),
        "upstream_commit": config.upstream_commit,
        "checkout_diff_sha256": actual_diff,
        "environment": json.loads(environment_path.read_text()),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": file_sha256(final_checkpoint),
        "model_bytes": final_checkpoint.stat().st_size,
        "gpu": gpu,
        "command": list(command),
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_path.write_text(_json(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export")
    train = commands.add_parser("train")
    train.add_argument("--arm", required=True)
    train.add_argument("--gpu", default="0")
    args = parser.parse_args()
    config = Pcgcv2RetrainConfig.from_json(args.config)
    result = (
        export_exact_split(config)
        if args.command == "export"
        else run_training_arm(config, args.arm, gpu=args.gpu)
    )
    print(_json(result), end="")


if __name__ == "__main__":
    main()
