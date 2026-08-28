#!/usr/bin/env bash
set -euo pipefail

# Run this on the EmpireAI login node after `empire_gpu.py sync` reports a live
# Jupyter allocation. The dispatcher preserves that allocation's GPU cgroup.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${POINTCONSTELLATION_PYTHON:-.venv/bin/python}"
minimum_vram_gib="${POINTCONSTELLATION_MIN_VRAM_GIB:-20}"
config="${POINTCONSTELLATION_EXPERIMENT_023_CONFIG:-configs/experiment_023_feature_codec_equal_protocol.json}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is unavailable: $python_bin" >&2
  exit 1
fi

exec "$python_bin" scripts/empire_gpu.py run \
  --min-vram "$minimum_vram_gib" \
  --desc "experiment-023 equal-protocol feature codec, six seeds" \
  -- "$python_bin" -m pointconstellation.feature_codec_benchmark \
  --config "$config" \
  --device cuda \
  --resume
