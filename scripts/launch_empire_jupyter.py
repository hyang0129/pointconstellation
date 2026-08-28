#!/usr/bin/env python3
"""Guarded EmpireAI Jupyter allocation launcher with no cancellation path."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.cluster.empire import (  # noqa: E402
    parse_all_jobs,
    pointconstellation_jobs,
    validate_launch,
)


def query_jobs(timeout: int = 20) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%N"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "squeue is unavailable; run this on the EmpireAI login node"
        ) from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "squeue failed")
    return parse_all_jobs(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submit after checks; without this flag the command is a dry run",
    )
    parser.add_argument(
        "--jupyter-script",
        default=os.environ.get(
            "EMPIRE_JUPYTER_SCRIPT",
            str(Path.home() / "rit_rc_scripts" / "empire_jupyter_lab.sh"),
        ),
    )
    args = parser.parse_args()

    try:
        jobs = query_jobs()
        validate_launch(jobs, args.port)
    except (RuntimeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(10) from exc

    script = Path(args.jupyter_script).expanduser()
    if not script.is_file():
        print(f"REFUSED: Jupyter batch script not found: {script}", file=sys.stderr)
        raise SystemExit(12)
    command = [
        "sbatch",
        f"--job-name=jupyter_empire_{args.port}",
        "--cpus-per-task=16",
        "--mem-per-cpu=24g",
        "--time=0-72:00:00",
        "--qos=rit",
        str(script),
        str(args.port),
    ]
    print(
        f"Current state: {len(pointconstellation_jobs(jobs))} Point "
        f"Constellation 86xx job(s); {len(jobs)} total user job(s).\n"
        f"Command: {shlex_join(command)}"
    )
    if not args.submit:
        print("Dry run only. Re-run with --submit after confirming the allocation.")
        return
    result = subprocess.run(command, check=False)
    raise SystemExit(result.returncode)


def shlex_join(parts: list[str]) -> str:
    import shlex

    return shlex.join(parts)


if __name__ == "__main__":
    main()
