#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/.." && pwd)
upstream_url="https://github.com/zb12138/OctAttention.git"
upstream_branch="obj"
upstream_commit="adb628b29abc4b160f55fe27dd43b0db7b730cac"
patch_path="$project_root/patches/octattention_lowrate.patch"
patch_sha256="81d081e63061b9cb5b922b2548fb346ab52912c4b4e8f76e1bb37e78961287f2"
checkout_diff_sha256="81d081e63061b9cb5b922b2548fb346ab52912c4b4e8f76e1bb37e78961287f2"
external_root="$project_root/artifacts/external/octattention"
conda_executable="${CONDA_EXE:-}"
create_env=0

usage() {
  cat <<'EOF'
Usage: scripts/prepare_octattention.sh [options]

Pins the OctAttention object checkpoint and applies the recorded low-rate batch
adapter. The PyTorch environment is opt-in and remains under ignored artifacts/.

Options:
  --root PATH    external workspace (default: artifacts/external/octattention)
  --conda PATH   conda executable for the isolated Python 3.10 environment
  --create-env   create and record the PyTorch/CUDA environment
  -h, --help     show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      external_root="$2"
      shift 2
      ;;
    --conda)
      conda_executable="$2"
      shift 2
      ;;
    --create-env)
      create_env=1
      shift
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

actual_patch_sha256=$(shasum -a 256 "$patch_path" | awk '{print $1}')
if [[ "$actual_patch_sha256" != "$patch_sha256" ]]; then
  echo "error: OctAttention patch SHA-256 differs" >&2
  exit 1
fi

mkdir -p "$external_root"
upstream_dir="$external_root/upstream"
if [[ ! -d "$upstream_dir/.git" ]]; then
  if [[ -e "$upstream_dir" ]]; then
    echo "error: $upstream_dir exists but is not a Git checkout" >&2
    exit 1
  fi
  git clone --branch "$upstream_branch" "$upstream_url" "$upstream_dir"
fi
git -C "$upstream_dir" fetch --depth 1 origin "$upstream_commit"
git -C "$upstream_dir" checkout --detach "$upstream_commit"
actual_commit=$(git -C "$upstream_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$upstream_commit" ]]; then
  echo "error: upstream commit verification failed" >&2
  exit 1
fi

current_diff_sha256=$(git -C "$upstream_dir" diff --binary HEAD | shasum -a 256 | awk '{print $1}')
empty_diff_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
if [[ "$current_diff_sha256" == "$empty_diff_sha256" ]]; then
  if [[ -e "$upstream_dir/pointconstellation_adapter.py" ]]; then
    echo "error: untracked OctAttention adapter prevents verified patching" >&2
    exit 1
  fi
  git -C "$upstream_dir" apply "$patch_path"
  git -C "$upstream_dir" add -N pointconstellation_adapter.py
elif [[ "$current_diff_sha256" != "$checkout_diff_sha256" ]]; then
  echo "error: OctAttention checkout contains an unregistered patch" >&2
  exit 1
fi
actual_diff_sha256=$(git -C "$upstream_dir" diff --binary HEAD | shasum -a 256 | awk '{print $1}')
if [[ "$actual_diff_sha256" != "$checkout_diff_sha256" ]]; then
  echo "error: applied OctAttention patch identity differs" >&2
  exit 1
fi

release_checkpoint="$upstream_dir/modelsave/obj/encoder_epoch_00800093.pth"
if [[ ! -s "$release_checkpoint" ]]; then
  echo "error: released object checkpoint is missing or empty" >&2
  exit 1
fi

if [[ "$create_env" -eq 1 ]]; then
  if [[ -z "$conda_executable" ]]; then
    conda_executable=$(command -v conda || true)
  fi
  if [[ -z "$conda_executable" || ! -x "$conda_executable" ]]; then
    echo "error: pass --conda PATH to an executable conda installation" >&2
    exit 1
  fi
  env_dir="$external_root/env"
  if [[ ! -x "$env_dir/bin/python" ]]; then
    "$conda_executable" create --yes --prefix "$env_dir" python=3.10 pip
  fi
  "$conda_executable" install --yes --prefix "$env_dir" \
    -c pytorch -c nvidia "pytorch=2.2.2" "pytorch-cuda=12.1"
  "$env_dir/bin/python" -m pip install \
    "numpy<2" "h5py<4" "hdf5storage<0.2" "plyfile<2" \
    "tensorboard<3" "tqdm<5"
  "$env_dir/bin/python" "$upstream_dir/pointconstellation_adapter.py" --help >/dev/null
  "$env_dir/bin/python" - \
    "$external_root/environment.json" \
    "$actual_commit" \
    "$actual_diff_sha256" <<'PY'
import json
import platform
import subprocess
import sys

import numpy
import torch

output, commit, diff_sha256 = sys.argv[1:]
record = {
    "python": sys.version,
    "platform": platform.platform(),
    "upstream_commit": commit,
    "checkout_diff_sha256": diff_sha256,
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available_at_prepare": torch.cuda.is_available(),
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

echo "Pinned OctAttention $upstream_branch at $actual_commit in $upstream_dir"
echo "Applied adapter diff SHA-256: $actual_diff_sha256"
if [[ "$create_env" -eq 0 ]]; then
  echo "Environment not created; add --create-env --conda PATH on the GPU host."
fi
