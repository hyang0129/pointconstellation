"""Run-once evaluator for the untouched Experiment 024 final slice."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pointconstellation.data import file_sha256, load_mesh_manifest
from pointconstellation.official_stability import (
    OfficialStabilityConfig,
    run_official_stability,
)


class FinalSliceAlreadyCompletedError(RuntimeError):
    """Raised when the immutable final slice already has completed metrics."""


@dataclass(frozen=True)
class FinalSliceConfig:
    """Configuration for the guarded Experiment 024 invocation."""

    final_slice_manifest: str
    official_stability_config: str
    output_root: str = "artifacts/local/final_slice"
    method_sources: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.final_slice_manifest:
            raise ValueError("final_slice_manifest must be nonempty")
        if not self.official_stability_config:
            raise ValueError("official_stability_config must be nonempty")
        if not self.output_root:
            raise ValueError("output_root must be nonempty")
        if any(not name or not source for name, source in self.method_sources.items()):
            raise ValueError("method_sources names and paths must be nonempty")

    @classmethod
    def from_json(cls, path: Path) -> FinalSliceConfig:
        return cls(**json.loads(path.read_text()))


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("git rev-parse did not return a full commit hash")
    return commit


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_or_validate_lock(path: Path, lock: dict[str, str]) -> None:
    try:
        with path.open("x") as handle:
            handle.write(json.dumps(lock, indent=2) + "\n")
    except FileExistsError:
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"existing final-slice lock is unreadable: {path}"
            ) from error
        if existing.get("manifest_sha256") != lock["manifest_sha256"]:
            raise RuntimeError(
                "existing final-slice lock names another manifest"
            ) from None
        if existing.get("git_commit") != lock["git_commit"]:
            raise RuntimeError(
                "refusing to resume final slice from another git commit"
            ) from None


def run_final_slice(
    config: FinalSliceConfig,
    *,
    device_name: str | None = None,
    pc_error_executable: Path | None = None,
) -> dict[str, Any]:
    """Run the official evaluator once for the exact final manifest bytes."""

    manifest_path = Path(config.final_slice_manifest)
    manifest_sha256 = file_sha256(manifest_path)
    output_dir = Path(config.output_root) / manifest_sha256
    metrics_path = output_dir / "official_metrics.json"
    if metrics_path.is_file():
        raise FinalSliceAlreadyCompletedError(
            f"final slice already completed for manifest {manifest_sha256}"
        )

    manifest = load_mesh_manifest(manifest_path)
    if set(manifest["splits"]) != {"final_validation", "final_ood"}:
        raise ValueError(
            "final-slice manifest must contain final_validation and final_ood only"
        )
    official = OfficialStabilityConfig.from_json(Path(config.official_stability_config))
    if official.max_clouds_per_split is not None:
        raise ValueError("final-slice evaluation forbids max_clouds_per_split")
    if pc_error_executable is not None:
        official = replace(
            official,
            pc_error_executable=str(pc_error_executable),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics_path.is_file():
        raise FinalSliceAlreadyCompletedError(
            f"final slice already completed for manifest {manifest_sha256}"
        )
    lock = {
        "manifest_sha256": manifest_sha256,
        "git_commit": _git_commit(),
        "timestamp": _timestamp(),
    }
    _write_or_validate_lock(output_dir / "FINAL_SLICE_LOCK", lock)
    official = replace(
        official,
        output_dir=str(output_dir),
        experiment_name="024_final_slice",
        evaluation_manifest=str(manifest_path),
        expected_evaluation_manifest_sha256=manifest_sha256,
        evaluation_split_map={
            "validation": "final_validation",
            "ood": "final_ood",
        },
    )
    return run_official_stability(official, device_name=device_name)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--pc-error", type=Path)
    parser.add_argument("--max-clouds-per-split", type=int)
    args = parser.parse_args(argv)
    if args.max_clouds_per_split is not None:
        parser.error("--max-clouds-per-split is forbidden for the final slice")
    config = FinalSliceConfig.from_json(args.config)
    run_final_slice(
        config,
        device_name=args.device,
        pc_error_executable=args.pc_error,
    )


if __name__ == "__main__":
    main()
