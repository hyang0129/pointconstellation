#!/usr/bin/env bash
set -euo pipefail

# Run this on the EmpireAI login node after `empire_gpu.py sync` reports live
# Jupyter allocations. Each dispatch owns one resumable (K, q) cell.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${POINTCONSTELLATION_PYTHON:-.venv/bin/python}"
minimum_vram_gib="${POINTCONSTELLATION_MIN_VRAM_GIB:-20}"
config="${POINTCONSTELLATION_EXPERIMENT_025_CONFIG:-configs/experiment_025_rate_sweep_modelnet40.json}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is unavailable: $python_bin" >&2
  exit 1
fi

constellation_sizes=(4 6 8 12 16)
coordinate_bits=(8 10 12)

if [[ -n "${POINTCONSTELLATION_RATE_CELL_K:-}" || -n "${POINTCONSTELLATION_RATE_CELL_Q:-}" ]]; then
  if [[ -z "${POINTCONSTELLATION_RATE_CELL_K:-}" || -z "${POINTCONSTELLATION_RATE_CELL_Q:-}" ]]; then
    echo "Set both POINTCONSTELLATION_RATE_CELL_K and POINTCONSTELLATION_RATE_CELL_Q." >&2
    exit 1
  fi
  constellation_sizes=("$POINTCONSTELLATION_RATE_CELL_K")
  coordinate_bits=("$POINTCONSTELLATION_RATE_CELL_Q")
fi

for constellation_size in "${constellation_sizes[@]}"; do
  for bits in "${coordinate_bits[@]}"; do
    "$python_bin" scripts/empire_gpu.py run \
      --min-vram "$minimum_vram_gib" \
      --desc "experiment-025 K=${constellation_size} q=${bits}" \
      -- "$python_bin" -m pointconstellation.rate_sweep_experiment \
      --config "$config" \
      --device cuda \
      --cell "$constellation_size" "$bits"
  done
done

echo "After every cell finishes, rebuild the curve with:"
echo "$python_bin -m pointconstellation.rate_sweep_experiment --config $config --aggregate-only"
