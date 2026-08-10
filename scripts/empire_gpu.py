#!/usr/bin/env python3
"""Discover EmpireAI Jupyter GPUs and dispatch tracked background jobs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.cluster.empire import (  # noqa: E402
    discover_allocations,
    expand_remote_path,
    sync_node_registry,
)
from pointconstellation.cluster.jupyter import JupyterExecutor  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "empire.nodes.json"


@dataclass(frozen=True)
class Node:
    name: str
    hostname: str
    jupyter_url: str
    python: str
    project_root: str
    max_concurrent_jobs: int
    kernel_name: str


@dataclass
class NodeHealth:
    reachable: bool = False
    gpu_name: str = ""
    used_mb: float = 0.0
    total_mb: float = 0.0
    utilization: float = 0.0
    error: str = ""

    @property
    def free_gb(self) -> float:
        return (self.total_mb - self.used_mb) / 1024.0


@dataclass
class Job:
    job_id: str
    node_name: str
    hostname: str
    command: str
    pid: int | None
    started_at: str
    status: str
    log_file: str
    description: str = ""


def load_config(path: Path) -> tuple[dict[str, Any], list[Node]]:
    if not path.exists():
        raise RuntimeError(
            f"missing {path}; copy configs/empire.nodes.example.json to "
            "configs/empire.nodes.json first"
        )
    raw = json.loads(path.read_text())
    defaults = raw.get("defaults", {})
    nodes = []
    for entry in raw.get("nodes", []):
        nodes.append(
            Node(
                name=entry["name"],
                hostname=entry["hostname"],
                jupyter_url=entry["jupyter_url"],
                python=expand_remote_path(
                    entry.get("python", defaults.get("python", "python3"))
                ),
                project_root=expand_remote_path(
                    entry.get("project_root", defaults.get("project_root", "."))
                ),
                max_concurrent_jobs=int(
                    entry.get(
                        "max_concurrent_jobs",
                        defaults.get("max_concurrent_jobs", 1),
                    )
                ),
                kernel_name=entry.get(
                    "kernel_name", defaults.get("kernel_name", "python3")
                ),
            )
        )
    return raw, nodes


def password_from_config(config: dict[str, Any]) -> str:
    variable = config.get("defaults", {}).get(
        "jupyter_password_env", "POINTCONSTELLATION_JUPYTER_PASSWORD"
    )
    password = os.environ.get(variable)
    if password is None:
        raise RuntimeError(f"set {variable} in the remote shell before connecting")
    return password


def manifest_path(config: dict[str, Any]) -> Path:
    defaults = config.get("defaults", {})
    root = Path(expand_remote_path(defaults.get("project_root", ".")))
    relative = defaults.get("job_manifest", "artifacts/empire/gpu_jobs.json")
    return root / relative


def read_jobs(path: Path) -> list[Job]:
    if not path.exists():
        return []
    with path.open() as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            records = json.load(handle)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return [Job(**record) for record in records]


def append_job(path: Path, job: Job) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        content = handle.read()
        records = json.loads(content) if content.strip() else []
        records.append(asdict(job))
        handle.seek(0)
        handle.truncate()
        json.dump(records, handle, indent=2)
        handle.write("\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def check_health(node: Node, password: str) -> NodeHealth:
    code = """import subprocess
r = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
     '--format=csv,noheader,nounits'], capture_output=True, text=True)
print(r.stdout.strip())
"""
    try:
        with JupyterExecutor(
            node.jupyter_url, password, kernel_name=node.kernel_name
        ) as executor:
            result = executor.run(code, timeout=30)
        if result.status != "ok" or not result.stdout.strip():
            return NodeHealth(error=result.error or "empty nvidia-smi response")
        first_line = result.stdout.strip().splitlines()[0]
        name, utilization, used, total = [
            value.strip() for value in first_line.split(",")[:4]
        ]
        return NodeHealth(
            reachable=True,
            gpu_name=name,
            utilization=float(utilization),
            used_mb=float(used),
            total_mb=float(total),
        )
    except Exception as exc:
        return NodeHealth(error=str(exc))


def health_for_nodes(nodes: list[Node], password: str) -> dict[str, NodeHealth]:
    statuses = {}
    with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        futures = {pool.submit(check_health, node, password): node for node in nodes}
        for future in as_completed(futures):
            node = futures[future]
            statuses[node.name] = future.result()
    return statuses


def cmd_sync(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        raise RuntimeError(
            "create configs/empire.nodes.json from the committed example first"
        )
    allocations = discover_allocations()
    summary = sync_node_registry(config_path, allocations, dry_run=args.dry_run)
    print(f"Found {len(allocations)} running Jupyter allocation(s).")
    for allocation in allocations:
        print(
            f"  {allocation.node_name} job={allocation.job_id} "
            f"time_left={allocation.time_left}"
        )
    for key, marker in (("added", "+"), ("removed", "-"), ("kept", "~")):
        for name in summary[key]:
            print(f"{marker} {name}")
    if args.dry_run:
        print("Dry run: registry not written.")


def cmd_status(args: argparse.Namespace) -> None:
    config, nodes = load_config(Path(args.config))
    if not nodes:
        print("No live nodes. Run the sync command after obtaining an allocation.")
        return
    password = password_from_config(config)
    statuses = health_for_nodes(nodes, password)
    jobs = read_jobs(manifest_path(config))
    running_counts = {
        node.name: sum(
            job.status == "running" and job.node_name == node.name for job in jobs
        )
        for node in nodes
    }
    print(f"{'NODE':<24} {'GPU':<22} {'FREE/TOTAL':<18} {'UTIL':<8} {'JOBS':<6}")
    for node in nodes:
        status = statuses[node.name]
        if not status.reachable:
            print(f"{node.name:<24} {'unreachable':<22} {status.error[:38]}")
            continue
        memory = f"{status.free_gb:.1f}/{status.total_mb / 1024:.1f} GB"
        print(
            f"{node.name:<24} {status.gpu_name[:21]:<22} {memory:<18} "
            f"{status.utilization:.0f}%{'':<4} {running_counts[node.name]:<6}"
        )


def choose_node(
    nodes: list[Node],
    statuses: dict[str, NodeHealth],
    jobs: list[Job],
    *,
    requested: str | None,
    min_vram: float,
) -> Node:
    candidates = []
    for node in nodes:
        if requested and node.name != requested:
            continue
        status = statuses[node.name]
        running = sum(
            job.status == "running" and job.node_name == node.name for job in jobs
        )
        if (
            status.reachable
            and status.free_gb >= min_vram
            and running < node.max_concurrent_jobs
        ):
            candidates.append((running, -status.free_gb, node))
    if not candidates:
        target = f" named {requested}" if requested else ""
        raise RuntimeError(
            f"no available Jupyter node{target} with at least {min_vram:.1f} GB free"
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
    return candidates[0][2]


def cmd_run(args: argparse.Namespace) -> None:
    config, nodes = load_config(Path(args.config))
    if not nodes:
        raise RuntimeError("no nodes in registry; run sync first")
    command_parts = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command_parts:
        raise RuntimeError("a command is required after --")
    command = shlex.join(command_parts)
    password = password_from_config(config)
    statuses = health_for_nodes(nodes, password)
    jobs_path = manifest_path(config)
    node = choose_node(
        nodes,
        statuses,
        read_jobs(jobs_path),
        requested=args.node,
        min_vram=args.min_vram,
    )

    job_id = uuid.uuid4().hex[:12]
    relative_log = f"artifacts/empire/logs/{job_id}.log"
    absolute_log = str(Path(node.project_root) / relative_log)
    inner_command = f"cd {shlex.quote(node.project_root)} && {command}"
    log_directory = str(Path(absolute_log).parent)
    code = (
        "import pathlib, subprocess\n"
        f"pathlib.Path({log_directory!r}).mkdir(parents=True, exist_ok=True)\n"
        f"_log = open({absolute_log!r}, 'w')\n"
        f"_process = subprocess.Popen(['bash', '-lc', {inner_command!r}], "
        "stdout=_log, stderr=_log, start_new_session=True)\n"
        "print(_process.pid)\n"
    )
    with JupyterExecutor(
        node.jupyter_url, password, kernel_name=node.kernel_name
    ) as executor:
        result = executor.run(code, timeout=30)
    if result.status != "ok":
        raise RuntimeError(result.error or result.stdout)
    try:
        pid = int(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        pid = None
    job = Job(
        job_id=job_id,
        node_name=node.name,
        hostname=node.hostname,
        command=command,
        pid=pid,
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running" if pid else "unknown",
        log_file=relative_log,
        description=args.desc,
    )
    append_job(jobs_path, job)
    print(json.dumps(asdict(job), indent=2))


def cmd_jobs(args: argparse.Namespace) -> None:
    config, _ = load_config(Path(args.config))
    jobs = read_jobs(manifest_path(config))
    if not args.all:
        jobs = [job for job in jobs if job.status == "running"]
    if not jobs:
        print("No tracked jobs.")
        return
    print(f"{'JOB':<14} {'NODE':<24} {'STATUS':<10} {'PID':<9} COMMAND")
    for job in jobs:
        print(
            f"{job.job_id:<14} {job.node_name:<24} {job.status:<10} "
            f"{str(job.pid or '-'):<9} {job.command}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    commands = parser.add_subparsers(dest="action", required=True)
    sync = commands.add_parser("sync", help="refresh Jupyter nodes from squeue")
    sync.add_argument("--dry-run", action="store_true")
    commands.add_parser("status", help="show GPU health and availability")
    jobs = commands.add_parser("jobs", help="show locally tracked jobs")
    jobs.add_argument("--all", action="store_true")
    run = commands.add_parser("run", help="dispatch a background command")
    run.add_argument("--node")
    run.add_argument("--min-vram", type=float, default=0.0)
    run.add_argument("--desc", default="")
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        {
            "sync": cmd_sync,
            "status": cmd_status,
            "run": cmd_run,
            "jobs": cmd_jobs,
        }[args.action](args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
