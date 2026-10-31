#!/usr/bin/env bash
set -euo pipefail

root=/root/aic_a100
code="$root/rematch_code/experiments/semifinal_a_doc"
run="$root/rematch_runs/r0"
weights="$root/replica_a/weights/clip-vit-base-patch32"
labels=/mnt/复赛/复赛测试集标签.csv
cd "$code"

python -u deliver.py --test-root "$root/rematch_data/test/test" \
    --official-weights "$weights" --checkpoint "$run/final_ema.pt" \
    --output "$run/cache_only" --save-cache "$run/four_view_cache.pt" \
    --cache-only --workers 8 > "$run/cache.log" 2>&1
python -u sweep_rematch_readout.py --cache "$run/four_view_cache.pt" \
    --config "$run/config.json" --output "$run/sweep" > "$run/sweep.log" 2>&1
cmp "$run/e1/pred_results.csv" "$run/sweep/e1_mid_final1.csv"
for prediction in "$run"/sweep/*.csv; do
    name="$(basename "$prediction" .csv)"
    python score_rematch.py --labels "$labels" --prediction "$prediction" \
        > "$run/sweep/$name.score.json"
done
echo REMATCH_SWEEP_COMPLETE
