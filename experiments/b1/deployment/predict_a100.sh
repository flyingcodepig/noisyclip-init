#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$HERE/../code"
PYTHON="${PYTHON:-/root/b1-env-20261027/bin/python}"
export CUDA_VISIBLE_DEVICES=0
PREFIX="${RUN_PREFIX:-a100_20261027}"
MODEL="$CODE/runs/${PREFIX}_stage2/model.pt"
OUT="$CODE/runs/${PREFIX}_predictions"
mkdir -p "$OUT"
"$PYTHON" "$HERE/check_checkpoint.py" "$CODE/runs/${PREFIX}_stage2/last.pt" "$CODE/train_config_a100_stage2.json" 2048 "$CODE"
"$PYTHON" "$CODE/build_cache_a100.py" --split test --cache-dir "$CODE/image_cache_test" --workers 12
"$PYTHON" "$CODE/predict_a100.py" --model "$MODEL" --output "$OUT"
"$PYTHON" "$CODE/readout_a100.py" --model "$MODEL" --base "$OUT" --output "$OUT/pred_results.csv"
"$PYTHON" "$HERE/verify_predictions.py" "$OUT/pred_results.csv"
"$PYTHON" "$CODE/package_results.py" --csv "$OUT/pred_results.csv" --zip "$OUT/pred_results.zip"
