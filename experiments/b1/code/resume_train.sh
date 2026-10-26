#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STAGE="${1:?Usage: bash resume_train.sh stage1|stage2}"
case "$STAGE" in
  stage1) CONFIG="$HERE/train_config_stage1.json"; UNTIL=4096 ;;
  stage2) CONFIG="$HERE/train_config.json"; UNTIL=2048 ;;
  *) echo 'Stage must be stage1 or stage2' >&2; exit 2 ;;
esac
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  "$HERE/engine/train_with_cutmix_four_gpu.py" --workspace "$HERE" \
  --config "$CONFIG" --run "$STAGE" --until "$UNTIL" --resume
