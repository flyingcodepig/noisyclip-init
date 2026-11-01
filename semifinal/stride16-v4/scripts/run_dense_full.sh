#!/usr/bin/env bash
set -euo pipefail
root=/root/aic_a100/eat_semifinal
python=/root/miniconda3/envs/myconda/bin/python
weights=/root/aic_a100/replica_a/weights/clip-vit-base-patch32
cd "$root"
mkdir -p runs/dense_full
trap 'echo FAILED >> runs/dense_full/status.txt' ERR
echo TRAINING > runs/dense_full/status.txt
sha256sum dense_src/*.py config.dense_full.json /root/aic_a100/replica_a/prepared/train_inputs.npz > runs/dense_full/input_hashes.txt
"$python" -u dense_src/train.py --prepared /root/aic_a100/replica_a/prepared --official-weights "$weights" --run "$root/runs/dense_full" --config "$root/config.dense_full.json" > runs/dense_full/train.log 2>&1
echo INFERENCE >> runs/dense_full/status.txt
"$python" -u dense_src/deliver.py --test-root /root/aic_a100/replica_a/data/test --official-weights "$weights" --checkpoint "$root/runs/dense_full/final_ema.pt" --output "$root/runs/dense_full/submission" --graph-mode e1 --batch 16 > runs/dense_full/inference.log 2>&1
"$python" - <<'PY'
import csv,zipfile
from pathlib import Path
r=Path('/root/aic_a100/eat_semifinal/runs/dense_full/submission')
p=r/'pred_results.csv'
with p.open() as f: rows=list(csv.reader(f))
assert len(rows)==49857 and len({x[0] for x in rows})==49857
assert all(len(x)==2 and len(x[1])==4 and x[1].isdigit() and 0<=int(x[1])<500 for x in rows)
with zipfile.ZipFile(r/'submission.zip') as z:
    assert z.namelist()==['pred_results.csv'] and z.read('pred_results.csv')==p.read_bytes()
PY
sha256sum runs/dense_full/final_ema.pt runs/dense_full/submission/pred_results.csv runs/dense_full/submission/submission.zip > runs/dense_full/output_hashes.txt
echo COMPLETE >> runs/dense_full/status.txt
