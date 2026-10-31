#!/usr/bin/env bash
set -euo pipefail

prepare_pid="${1:?Pass the active prepare.py PID}"
root=/root/aic_a100
code="$root/rematch_code/experiments/semifinal_a_doc"
prepared="$root/rematch_prepared/r0"
run="$root/rematch_runs/r0"
weights="$root/replica_a/weights/clip-vit-base-patch32"
labels=/mnt/复赛/复赛测试集标签.csv
test_root="$root/rematch_data/test/test"

cd "$code"
mkdir -p "$run"
cp config_rematch.json "$run/config.json"
sha256sum "$run/config.json" "$weights/pytorch_model.bin" "$labels" > "$run/input_hashes.txt"

until grep -q '"images": 148695' "$root/rematch_prepare_r0.log"; do
    if ! kill -0 "$prepare_pid" 2>/dev/null; then
        echo "prepare.py stopped without a complete result" >&2
        exit 1
    fi
    sleep 30
done
test -s "$prepared/train_inputs.npz"

python -u train.py --prepared "$prepared" --official-weights "$weights" \
    --run "$root/rematch_runs/smoke" --config "$run/config.json" \
    --workers 8 --stop-after 2 > "$run/smoke.log" 2>&1
python -u train.py --prepared "$prepared" --official-weights "$weights" \
    --run "$run" --config "$run/config.json" --workers 8 > "$run/train.log" 2>&1
test -s "$run/final_ema.pt"

for mode in e1 joint; do
    output="$run/$mode"
    python -u deliver.py --test-root "$test_root" --official-weights "$weights" \
        --checkpoint "$run/final_ema.pt" --output "$output" --graph-mode "$mode" \
        --workers 8 > "$run/deliver_$mode.log" 2>&1
    python score_rematch.py --labels "$labels" --prediction "$output/pred_results.csv" \
        > "$run/score_${mode}_dev.json"
    sha256sum "$output/pred_results.csv" "$output/submission.zip" \
        > "$run/hashes_$mode.txt"
done
sha256sum "$run/final_ema.pt" "$prepared/train_inputs.npz" > "$run/output_hashes.txt"
echo REMATCH_BASELINE_COMPLETE
