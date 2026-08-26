#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: scripts/launch_pcgcv2_empire.sh ARM [NODE]" >&2
  exit 2
fi

arm="$1"
node="${2:-}"
command=(
  .venv/bin/python scripts/empire_gpu.py run
  --min-vram 20
  --desc "experiment-027 PCGCv2 exact retrain $arm"
)
if [[ -n "$node" ]]; then
  command+=(--node "$node")
fi
command+=(
  -- .venv/bin/python -m pointconstellation.pcgcv2_training
  --config configs/experiment_027_pcgcv2_retrain.json
  train --arm "$arm" --gpu 0
)
"${command[@]}"
