#!/usr/bin/env bash
set -euo pipefail

python_bin="${POINTCONSTELLATION_PYTHON:-.venv/bin/python}"

"${python_bin}" scripts/empire_gpu.py run \
  --min-vram 20 \
  --desc "experiment-028 Thingi10K stability" \
  -- "${python_bin}" -m pointconstellation.stability_experiment \
  --config configs/experiment_028_thingi10k_stability.json \
  --device cuda
