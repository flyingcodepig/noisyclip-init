#!/usr/bin/env bash
set -euo pipefail
root=/root/aic_a100/eat_semifinal
python=/root/miniconda3/envs/myconda/bin/python
weights=/root/aic_a100/replica_a/weights/clip-vit-base-patch32
cd "$root"
trap 'echo FAILED >> status.txt' ERR
echo SPLITTING > status.txt
"$python" -u bootstrap.py > split.log 2>&1
count=$("$python" -c 'import json;print(json.load(open("config.baseline.json"))["expected_train"])')
echo PREPARING >> status.txt
"$python" -u src/prepare.py --train-root "$root/split/train" --official-weights "$weights" --output "$root/prepared" --classes 500 --expected-train "$count" > prepare.log 2>&1
mkdir -p runs/baseline
sha256sum src/*.py config.baseline.json split/manifest.json "$weights/pytorch_model.bin" > runs/baseline/input_hashes.txt
echo SMOKE >> status.txt
"$python" -u src/train.py --prepared "$root/prepared" --official-weights "$weights" --run "$root/runs/smoke" --config "$root/config.baseline.json" --stop-after 2 > smoke.log 2>&1
"$python" - <<'PY'
import torch
s=torch.load('runs/smoke/last.pt',map_location='cpu',weights_only=False)
assert s['step']==2
assert all(torch.isfinite(v).all() for v in s['model'].values() if torch.is_floating_point(v))
PY
echo TRAINING >> status.txt
"$python" -u src/train.py --prepared "$root/prepared" --official-weights "$weights" --run "$root/runs/baseline" --config "$root/config.baseline.json" > train.log 2>&1
echo VALIDATING >> status.txt
"$python" -u src/deliver.py --test-root "$root/split/validation" --official-weights "$weights" --checkpoint "$root/runs/baseline/final_ema.pt" --output "$root/runs/baseline/validation" --graph-mode e1 > validation.log 2>&1
"$python" evaluate.py > runs/baseline/evaluation.txt
sha256sum runs/baseline/final_ema.pt runs/baseline/validation/pred_results.csv runs/baseline/validation_score.json > runs/baseline/output_hashes.txt
echo BASELINE_COMPLETE >> status.txt
