#!/usr/bin/env bash
set -euo pipefail

python_bin="${POINTCONSTELLATION_PYTHON:-.venv/bin/python}"

"${python_bin}" scripts/empire_gpu.py run \
  --min-vram 20 \
  --desc "experiment-029 ScanObjectNN stability" \
  -- "${python_bin}" -m pointconstellation.stability_experiment \
  --config configs/experiment_029_scanobjectnn_stability.json \
  --device cuda
