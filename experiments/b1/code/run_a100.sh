#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/root/b1-env-20261027/bin/python}"
export CUDA_VISIBLE_DEVICES=0
RUN_PREFIX="${RUN_PREFIX:-a100_20261027}"
mkdir -p "$HERE/runs"
exec 9>"$HERE/runs/.a100.lock"
flock -n 9 || { echo 'Another A100 pipeline holds the lock'; exit 2; }
run_stage() {
  local stage="$1" target="$2"
  shift 2
  local run="${RUN_PREFIX}_${stage}" cfg="$HERE/train_config_a100_${stage}.json"
  local checkpoint="$HERE/runs/$run/last.pt"
  if [[ -f "$checkpoint" ]]; then
    "$PYTHON" "$HERE/engine/train_single_gpu.py" --workspace "$HERE" --config "$cfg" --run "$run" --until "$target" --resume
  else
    "$PYTHON" "$HERE/engine/train_single_gpu.py" --workspace "$HERE" --config "$cfg" --run "$run" --until "$target" "$@"
  fi
  "$PYTHON" "$HERE/../deployment/check_checkpoint.py" "$checkpoint" "$cfg" "$target" "$HERE"
}
run_stage stage1 4096
run_stage stage2 1536 --restart-from "$HERE/runs/${RUN_PREFIX}_stage1/last.pt"
run_stage stage2 2048

