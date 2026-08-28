#!/usr/bin/env bash
set -euo pipefail

upstream_url="https://github.com/NJUVISION/PCGCv2.git"
upstream_commit="88ff2a18b1b3cac89eef66997cc4e8bcf4fb0420"
external_root="artifacts/external/pcgcv2"
environment_python=""

usage() {
  cat <<'EOF'
Usage: scripts/prepare_pcgcv2.sh [options]

Pins PCGCv2 and expands its released checkpoints. The CUDA/MinkowskiEngine
environment remains isolated from Point Constellation; pass --python to verify
an already-created environment and record its exact package inventory.

Options:
  --root PATH      external workspace (default: artifacts/external/pcgcv2)
  --python PATH    PCGCv2 Python with torch, MinkowskiEngine, and torchac
  -h, --help       show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      external_root="$2"
      shift 2
      ;;
    --python)
      environment_python="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$external_root"
upstream_dir="$external_root/upstream"
if [[ ! -d "$upstream_dir/.git" ]]; then
  if [[ -e "$upstream_dir" ]]; then
    echo "error: $upstream_dir exists but is not a Git checkout" >&2
    exit 1
  fi
  git clone "$upstream_url" "$upstream_dir"
fi
git -C "$upstream_dir" fetch --depth 1 origin "$upstream_commit"
git -C "$upstream_dir" checkout --detach "$upstream_commit"
actual_commit=$(git -C "$upstream_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$upstream_commit" ]]; then
  echo "error: PCGCv2 checkout verification failed" >&2
  exit 1
fi
if [[ -f "$upstream_dir/ckpts.zip" ]]; then
  unzip -n "$upstream_dir/ckpts.zip" -d "$upstream_dir"
fi
if [[ ! -f "$upstream_dir/ckpts/r3_0.10bpp.pth" ]]; then
  echo "error: released PCGCv2 checkpoint was not expanded as expected" >&2
  exit 1
fi
chmod u+x "$upstream_dir/tmc3"

if [[ -n "$environment_python" ]]; then
  if [[ ! -x "$environment_python" ]]; then
    echo "error: PCGCv2 Python is missing or not executable" >&2
    exit 1
  fi
  mkdir -p "$external_root/env/bin"
  ln -sfn "$(cd "$(dirname "$environment_python")" && pwd)/$(basename "$environment_python")" \
    "$external_root/env/bin/python"
  "$environment_python" - "$external_root/environment.json" "$actual_commit" <<'PY'
import json
import platform
import subprocess
import sys

import MinkowskiEngine as ME
import torch
import torchac

output, commit = sys.argv[1:]
if not torch.cuda.is_available():
    raise RuntimeError("PCGCv2 environment cannot see a CUDA GPU")
record = {
    "python": sys.version,
    "platform": platform.platform(),
    "upstream_commit": commit,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "minkowski_engine": getattr(ME, "__version__", "unknown"),
    "torchac": getattr(torchac, "__version__", "0.9.3"),
    "gpu": torch.cuda.get_device_name(0),
    "pip_freeze": subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines(),
}
with open(output, "w") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY
fi

echo "Pinned PCGCv2 at $actual_commit in $upstream_dir"
if [[ -z "$environment_python" ]]; then
  echo "Environment not verified; rerun with --python PATH on an allocated GPU."
fi
