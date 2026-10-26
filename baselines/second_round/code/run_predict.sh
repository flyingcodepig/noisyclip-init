set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
PACKAGE_ROOT="$(cd "$HERE/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$PACKAGE_ROOT/AIC-2026-12985846-模型文件/model.pt}"
RESULT_DIR="${RESULT_DIR:-$HERE/../AIC-2026-12985846-赛题结果文档}"
INFERENCE_CACHE="${INFERENCE_CACHE:-$HERE/inference_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ -f "$MODEL_PATH" ]] || { echo "Model not found: $MODEL_PATH" >&2; exit 2; }
[[ -d "$HERE/data/official/test" ]] || { echo "Official TEST images not found: $HERE/data/official/test" >&2; exit 2; }

mkdir -p "$RESULT_DIR"
MODEL_PATH="$(realpath "$MODEL_PATH")"
RESULT_DIR="$(realpath "$RESULT_DIR")"
mkdir -p "$INFERENCE_CACHE"
INFERENCE_CACHE="$(realpath "$INFERENCE_CACHE")"
cd "$HERE"
"$PYTHON" build_cache.py --split test
"$PYTHON" predict.py --model "$MODEL_PATH" --output "$INFERENCE_CACHE"
"$PYTHON" readout.py --base "$INFERENCE_CACHE" --output "$RESULT_DIR/pred_results.csv" --model "$MODEL_PATH"
"$PYTHON" package_results.py --csv "$RESULT_DIR/pred_results.csv" --zip "$RESULT_DIR/pred_results.zip"

[[ -f "$RESULT_DIR/pred_results.csv" ]] || { echo "Prediction CSV not found: $RESULT_DIR/pred_results.csv" >&2; exit 2; }
[[ -f "$RESULT_DIR/pred_results.zip" ]] || { echo "Prediction ZIP not found: $RESULT_DIR/pred_results.zip" >&2; exit 2; }

echo "Inference completed."
echo "Result CSV: $RESULT_DIR/pred_results.csv"
echo "Result ZIP: $RESULT_DIR/pred_results.zip"
