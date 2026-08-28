#!/usr/bin/env bash
set -euo pipefail

upstream_url="https://github.com/mauriceqch/pcc_geo_cnn_v2.git"
upstream_commit="b7a4ae2a548ad3c44a04af139dd77d804cf3a6fa"
record_url="https://zenodo.org/api/records/18507162/files/pcc_geo_cnn_v2.zip/content"
archive_md5="dc930e34c2ad1f8a360b80a313df9526"
external_root="artifacts/external/pcc_geo_cnn_v2"
conda_executable="${CONDA_EXE:-}"
create_env=0
download_artifacts=0

usage() {
  cat <<'EOF'
Usage: scripts/prepare_pcc_geo_cnn_v2.sh [options]

Pins the official pcc_geo_cnn_v2 checkout. Large release artifacts and the
legacy Python environment are opt-in and remain under ignored artifacts/.

Options:
  --root PATH            external workspace (default: artifacts/external/...)
  --conda PATH           conda executable for the Python 3.7 environment
  --create-env           create the isolated TensorFlow 1.15 GPU environment
  --download-artifacts   download and verify the 5.5 GB Zenodo release
  -h, --help             show this help
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
    --download-artifacts)
      download_artifacts=1
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
  echo "error: upstream commit verification failed" >&2
  exit 1
fi

if [[ "$download_artifacts" -eq 1 ]]; then
  archive="$external_root/pcc_geo_cnn_v2.zip"
  if [[ ! -f "$archive" ]]; then
    curl --fail --location --continue-at - --output "$archive" "$record_url"
  fi
  actual_md5=$(python3 - "$archive" <<'PY'
import hashlib
import sys

digest = hashlib.md5(usedforsecurity=False)
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
  )
  if [[ "$actual_md5" != "$archive_md5" ]]; then
    echo "error: Zenodo archive checksum mismatch" >&2
    exit 1
  fi
  release_dir="$external_root/release"
  mkdir -p "$release_dir"
  unzip -n "$archive" -d "$release_dir"
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
    "$conda_executable" create --yes --prefix "$env_dir" python=3.7 pip
  fi
  "$conda_executable" install --yes --prefix "$env_dir" \
    "cudatoolkit=10.0.130" "cudnn=7.6.5"
  "$env_dir/bin/python" -m pip install --upgrade "pip<24" "setuptools<69"
  "$env_dir/bin/python" -m pip install -r "$upstream_dir/requirements.txt"
  # TensorFlow 1.15's generated protobuf bindings predate the protobuf 4.x
  # runtime that pip otherwise resolves in a fresh environment.
  "$env_dir/bin/python" -m pip uninstall --yes tensorflow tensorflow-gpu
  "$env_dir/bin/python" -m pip install \
    "tensorflow-gpu==1.15.5" "protobuf==3.20.3"
  export LD_LIBRARY_PATH="$env_dir/lib:${LD_LIBRARY_PATH:-}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  "$env_dir/bin/python" - "$external_root/environment.json" "$actual_commit" <<'PY'
import json
import platform
import subprocess
import sys

import tensorflow as tf
import tensorflow_compression as tfc

output, commit = sys.argv[1:]
record = {
    "python": sys.version,
    "platform": platform.platform(),
    "upstream_commit": commit,
    "tensorflow": tf.__version__,
    "tensorflow_compression": getattr(tfc, "__version__", "1.3"),
    "gpu_available": bool(tf.test.is_gpu_available(cuda_only=True)),
    "pip_freeze": subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines(),
}
if not record["gpu_available"]:
    raise RuntimeError("TensorFlow 1.15 cannot see a compatible GPU")
with open(output, "w") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY
fi

echo "Pinned pcc_geo_cnn_v2 at $actual_commit in $upstream_dir"
if [[ "$download_artifacts" -eq 0 ]]; then
  echo "Release artifacts not downloaded; add --download-artifacts when ready."
fi
if [[ "$create_env" -eq 0 ]]; then
  echo "Legacy environment not created; add --create-env --conda PATH on Linux."
fi
