#!/usr/bin/env bash
set -euo pipefail

stage=""
node=""
seed=""
smoke=0
min_vram=20
nodes_config="configs/empire.nodes.json"
codec_config="configs/external/octattention_lowrate.json"
controller_python=".venv-train/bin/python"

usage() {
  cat <<'EOF'
Usage: scripts/launch_octattention_empire.sh STAGE [options]

Dispatch one Experiment 026 stage through the tracked EmpireAI Jupyter runner.
Run stages in order: export, train, evaluate. The script does not submit a
cluster run unless invoked explicitly.

Stages:
  export      export the exact 512 Experiment 019 training sources
  train       train seeds 7, 17, and 29 (or one seed selected with --seed)
  evaluate    run both explicitly labeled arms at depths 4, 5, 6, and 7

Options:
  --node NAME             request one registered logical node
  --seed N                train only seed 7, 17, or 29
  --smoke                 use two training steps or two clouds per split
  --min-vram GB           required free VRAM (default: 20)
  --nodes-config PATH     EmpireAI node registry
  --codec-config PATH     Experiment 026 config
  --python PATH           controller Python on the remote project
  -h, --help              show this help
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi
stage="$1"
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node)
      node="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --smoke)
      smoke=1
      shift
      ;;
    --min-vram)
      min_vram="$2"
      shift 2
      ;;
    --nodes-config)
      nodes_config="$2"
      shift 2
      ;;
    --codec-config)
      codec_config="$2"
      shift 2
      ;;
    --python)
      controller_python="$2"
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

if [[ "$stage" != "export" && "$stage" != "train" && "$stage" != "evaluate" ]]; then
  echo "error: stage must be export, train, or evaluate" >&2
  exit 2
fi
if [[ -n "$seed" && "$seed" != "7" && "$seed" != "17" && "$seed" != "29" ]]; then
  echo "error: --seed must be 7, 17, or 29" >&2
  exit 2
fi

dispatch() {
  local description="$1"
  shift
  local command=(
    "$controller_python" scripts/empire_gpu.py
    --config "$nodes_config" run
    --min-vram "$min_vram"
    --desc "$description"
  )
  if [[ -n "$node" ]]; then
    command+=(--node "$node")
  fi
  command+=(-- "$@")
  "${command[@]}"
}

if [[ "$stage" == "export" ]]; then
  dispatch \
    "experiment-026 exact training export" \
    "$controller_python" -m pointconstellation.octattention_benchmark \
    --config "$codec_config" --export-training-sources
elif [[ "$stage" == "train" ]]; then
  seeds=(7 17 29)
  if [[ -n "$seed" ]]; then
    seeds=("$seed")
  fi
  for training_seed in "${seeds[@]}"; do
    checkpoint="artifacts/external/octattention/checkpoints/retrained_seed_${training_seed}/encoder_final.pth"
    steps=100000
    if [[ "$smoke" -eq 1 ]]; then
      checkpoint="artifacts/external/octattention/checkpoints/smoke_seed_${training_seed}/encoder_final.pth"
      steps=2
    fi
    dispatch \
      "experiment-026 retrain seed ${training_seed}" \
      artifacts/external/octattention/env/bin/python \
      artifacts/external/octattention/upstream/pointconstellation_adapter.py train \
      --checkpoint "$checkpoint" \
      --training-manifest artifacts/external/octattention/training/training_manifest.json \
      --source-root artifacts/external/octattention/training/sources \
      --position-bits 12 --depths 4 5 6 7 \
      --seed "$training_seed" --max-steps "$steps" \
      --batch-size 32 --bptt 1024 --learning-rate 0.001
  done
else
  evaluation_command=(
    "$controller_python" -m pointconstellation.octattention_benchmark
    --config "$codec_config"
  )
  if [[ "$smoke" -eq 1 ]]; then
    evaluation_command+=(--max-clouds-per-split 2)
  fi
  dispatch "experiment-026 OctAttention evaluation" "${evaluation_command[@]}"
fi
