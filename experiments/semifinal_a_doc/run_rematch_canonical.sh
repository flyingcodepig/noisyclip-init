#!/usr/bin/env bash
set -euo pipefail

root=/root/aic_a100
code="$root/rematch_code/experiments/semifinal_a_doc"
prepared="$root/rematch_prepared/r0"
run="$root/rematch_runs/r1_canonical"
weights="$root/replica_a/weights/clip-vit-base-patch32"
labels=/mnt/复赛/复赛测试集标签.csv
cd "$code"
mkdir -p "$run"
cp config_rematch_canonical.json "$run/config.json"
sha256sum "$run/config.json" "$prepared/train_inputs.npz" \
    "$weights/pytorch_model.bin" > "$run/input_hashes.txt"

python -u train.py --prepared "$prepared" --official-weights "$weights" \
    --run "$root/rematch_runs/r1_canonical_smoke" --config "$run/config.json" \
    --workers 8 --stop-after 2 > "$run/smoke.log" 2>&1
python -u train.py --prepared "$prepared" --official-weights "$weights" \
    --run "$run" --config "$run/config.json" --workers 8 > "$run/train.log" 2>&1
python -u deliver.py --test-root "$root/rematch_data/test/test" \
    --official-weights "$weights" --checkpoint "$run/final_ema.pt" \
    --output "$run/e1" --graph-mode e1 --workers 8 > "$run/deliver_e1.log" 2>&1
python score_rematch.py --labels "$labels" \
    --prediction "$run/e1/pred_results.csv" > "$run/score_e1_dev.json"
sha256sum "$run/final_ema.pt" "$run/e1/pred_results.csv" \
    "$run/e1/submission.zip" > "$run/output_hashes.txt"
echo CANONICAL_RUN_COMPLETE
