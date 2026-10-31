#!/usr/bin/env bash
set -euo pipefail

root=/root/aic_a100
code="$root/rematch_code/experiments/semifinal_a_doc"
run="$root/rematch_runs/r0"
python=/root/miniconda3/envs/myconda/bin/python
labels=/mnt/复赛/复赛测试集标签.csv
cd "$code"

for epsilon in 0.1 0.25; do
    name="ptmap_epsilon_${epsilon}"
    output="$run/$name"
    "$python" -u ptmap_rematch.py --cache "$run/view_cache.pt" \
        --config "$run/config.json" --output "$output" \
        --epsilon "$epsilon" --alpha 0.2 --iterations 20 \
        > "$run/$name.log" 2>&1
    "$python" score_rematch.py --labels "$labels" \
        --prediction "$output/pred_results.csv" > "$output/score_dev.json"
    sha256sum "$output/pred_results.csv" "$output/record.json" \
        > "$output/output_hashes.txt"
done
echo PTMAP_RUN_COMPLETE
