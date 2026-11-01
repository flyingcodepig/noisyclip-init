#!/usr/bin/env bash
set -euo pipefail
root=/root/aic_a100/eat_semifinal
python=/root/miniconda3/envs/myconda/bin/python
weights=/root/aic_a100/replica_a/weights/clip-vit-base-patch32
cd "$root"
mkdir -p runs/dense
trap 'echo FAILED >> runs/dense/status.txt' ERR
echo SMOKE > runs/dense/status.txt
"$python" -u dense_src/train.py --prepared "$root/prepared" --official-weights "$weights" --run "$root/runs/dense_smoke" --config "$root/config.dense.json" --stop-after 2 > runs/dense/smoke.log 2>&1
"$python" - <<'PY'
import torch
s=torch.load('runs/dense_smoke/last.pt',map_location='cpu',weights_only=False)
assert s['step']==2
assert all(torch.isfinite(v).all() for v in s['model'].values() if torch.is_floating_point(v))
PY
echo TRAINING >> runs/dense/status.txt
sha256sum dense_src/*.py config.dense.json split/manifest.json > runs/dense/input_hashes.txt
"$python" -u dense_src/train.py --prepared "$root/prepared" --official-weights "$weights" --run "$root/runs/dense" --config "$root/config.dense.json" > runs/dense/train.log 2>&1
echo VALIDATING >> runs/dense/status.txt
"$python" -u dense_src/deliver.py --test-root "$root/split/validation" --official-weights "$weights" --checkpoint "$root/runs/dense/final_ema.pt" --output "$root/runs/dense/validation" --graph-mode e1 --batch 16 > runs/dense/validation.log 2>&1
"$python" - <<'PY'
import csv,json,hashlib
from pathlib import Path
r=Path('/root/aic_a100/eat_semifinal')
p=r/'runs/dense/validation/pred_results.csv'
rows=list(csv.reader(p.open()))
pred=dict(rows); truth=json.loads((r/'split/validation_labels.json').read_text())
assert len(rows)==len(pred)==len(truth)==9032 and set(pred)==set(truth)
correct=sum(pred[k]==v for k,v in truth.items())
out={'metric':'held_out_noisy_label_accuracy_not_official_score','correct':correct,'total':len(truth),'accuracy':correct/len(truth),'prediction_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(r/'runs/dense/validation_score.json').write_text(json.dumps(out,indent=2))
print(out)
PY
sha256sum runs/dense/final_ema.pt runs/dense/validation/pred_results.csv runs/dense/validation_score.json > runs/dense/output_hashes.txt
echo COMPLETE >> runs/dense/status.txt
