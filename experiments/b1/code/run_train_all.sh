set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FINAL_MODEL="$HERE/runs/stage2/model.pt"
RESULT_DIR="${RESULT_DIR:-$HERE/../results}"

bash "$HERE/train_stage1.sh"
bash "$HERE/train_stage2.sh"

[[ -f "$FINAL_MODEL" ]] || { echo "Final checkpoint not found: $FINAL_MODEL" >&2; exit 2; }

MODEL_PATH="$FINAL_MODEL" \
RESULT_DIR="$RESULT_DIR" \
bash "$HERE/run_predict.sh"

[[ -f "$RESULT_DIR/pred_results.csv" ]] || { echo "Prediction CSV not found: $RESULT_DIR/pred_results.csv" >&2; exit 2; }
[[ -f "$RESULT_DIR/pred_results.zip" ]] || { echo "Prediction ZIP not found: $RESULT_DIR/pred_results.zip" >&2; exit 2; }

echo "Reproduction completed."
echo "Final model: $FINAL_MODEL"
echo "Result CSV: $RESULT_DIR/pred_results.csv"
echo "Result ZIP: $RESULT_DIR/pred_results.zip"
