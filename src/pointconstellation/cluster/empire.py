"""SLURM discovery and guarded EmpireAI allocation helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PORT_MIN = 8600
PORT_MAX = 8699
MAX_RUNNING_JUPYTER = 6
MAX_TOTAL_JOBS = 6
JUPYTER_JOB_RE = re.compile(r"^jupyter_[A-Za-z0-9_-]+_(\d+)$")


def _job_port(name: str) -> int | None:
    match = JUPYTER_JOB_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def is_pointconstellation_job(name: str) -> bool:
    """Return whether a job occupies the reserved Point Constellation namespace."""

    port = _job_port(name)
    return port is not None and PORT_MIN <= port <= PORT_MAX


def pointconstellation_jobs(
    jobs: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Keep only jobs in the reserved 86xx workstream."""

    return [job for job in jobs if is_pointconstellation_job(job["name"])]


def validate_logical_node(name: str, hostname: str, jupyter_url: str) -> int:
    """Validate a registry row and return its reserved Jupyter port."""

    parsed = urlsplit(jupyter_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid Jupyter URL for node {name}: {jupyter_url}") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or port is None:
        raise ValueError(f"invalid Jupyter URL for node {name}: {jupyter_url}")
    if parsed.hostname != hostname:
        raise ValueError(
            f"logical node {name} URL host {parsed.hostname} differs from {hostname}"
        )
    if not PORT_MIN <= port <= PORT_MAX:
        raise ValueError(
            f"logical node {name} must use a Point Constellation 86xx port"
        )
    expected_name = f"{hostname}-{port}"
    if name != expected_name:
        raise ValueError(f"logical node name must be {expected_name}, received {name}")
    return port


@dataclass(frozen=True)
class SlurmAllocation:
    """One running Jupyter allocation discovered through ``squeue``."""

    job_id: str
    name: str
    state: str
    hostname: str
    port: int
    time_left: str = ""

    @property
    def node_name(self) -> str:
        return f"{self.hostname}-{self.port}"

    @property
    def jupyter_url(self) -> str:
        return f"http://{self.hostname}:{self.port}"


def parse_squeue_allocations(output: str) -> list[SlurmAllocation]:
    """Parse ``%i|%j|%T|%N|%L`` rows, keeping usable Jupyter jobs."""

    allocations = []
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split("|")]
        if len(parts) < 4:
            continue
        job_id, name, state, hostname = parts[:4]
        time_left = parts[4] if len(parts) > 4 else ""
        port = _job_port(name)
        if state != "RUNNING" or port is None:
            continue
        if not PORT_MIN <= port <= PORT_MAX:
            continue
        if not hostname or "[" in hostname or "," in hostname:
            continue
        allocations.append(
            SlurmAllocation(job_id, name, state, hostname, port, time_left)
        )
    return allocations


def discover_allocations(timeout: int = 20) -> list[SlurmAllocation]:
    """Discover allocations on an EmpireAI login node."""

    try:
        result = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%N|%L"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "squeue is unavailable; run this command on the EmpireAI login node"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"squeue timed out after {timeout} seconds") from exc

    if result.returncode:
        raise RuntimeError(
            f"squeue failed with exit {result.returncode}: {result.stderr.strip()}"
        )
    return parse_squeue_allocations(result.stdout)


def expand_remote_path(value: str) -> str:
    """Expand ``~`` and environment variables on the login node."""

    return os.path.expandvars(os.path.expanduser(value))


def sync_node_registry(
    config_path: Path,
    allocations: list[SlurmAllocation],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Replace generated nodes with the current live SLURM allocations."""

    config = json.loads(config_path.read_text())
    defaults = config.get("defaults", {})
    old_names = {
        node.get("name", node.get("hostname", "")) for node in config.get("nodes", [])
    }
    nodes = [
        {
            "name": allocation.node_name,
            "hostname": allocation.hostname,
            "jupyter_url": allocation.jupyter_url,
            "slurm_job_id": allocation.job_id,
            "slurm_time_left": allocation.time_left,
            "python": defaults.get("python", "python3"),
            "project_root": defaults.get("project_root", "."),
            "max_concurrent_jobs": defaults.get("max_concurrent_jobs", 1),
            "kernel_name": defaults.get("kernel_name", "python3"),
        }
        for allocation in allocations
    ]
    new_names = {node["name"] for node in nodes}
    config["nodes"] = nodes
    if not dry_run:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    return {
        "added": sorted(new_names - old_names),
        "removed": sorted(old_names - new_names),
        "kept": sorted(old_names & new_names),
        "nodes": nodes,
        "written": not dry_run,
    }


def parse_all_jobs(output: str) -> list[dict[str, str]]:
    """Parse ``%i|%j|%T|%N`` rows for guarded-launch policy checks."""

    jobs = []
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split("|")]
        if len(parts) >= 4:
            jobs.append(
                dict(
                    zip(
                        ("job_id", "name", "state", "hostname"),
                        parts[:4],
                        strict=True,
                    )
                )
            )
    return jobs


def validate_launch(jobs: list[dict[str, str]], port: int) -> None:
    """Enforce the six-allocation Point Constellation 86xx namespace."""

    if not PORT_MIN <= port <= PORT_MAX:
        raise ValueError(f"port must be between {PORT_MIN} and {PORT_MAX}")
    project_jobs = pointconstellation_jobs(jobs)
    running = [job for job in project_jobs if job["state"] == "RUNNING"]
    if len(running) >= MAX_RUNNING_JUPYTER:
        raise RuntimeError(
            f"refusing launch: {len(running)} Jupyter jobs already running "
            f"(cap {MAX_RUNNING_JUPYTER})"
        )
    if len(project_jobs) >= MAX_TOTAL_JOBS:
        raise RuntimeError(
            f"refusing launch: {len(project_jobs)} Point Constellation jobs "
            f"already queued or running (cap {MAX_TOTAL_JOBS})"
        )
    for job in running:
        if _job_port(job["name"]) == port:
            raise RuntimeError(f"refusing launch: port {port} is already allocated")


def allocation_as_dict(allocation: SlurmAllocation) -> dict[str, Any]:
    """Return a JSON-safe allocation representation."""

    return {**asdict(allocation), "node_name": allocation.node_name}
