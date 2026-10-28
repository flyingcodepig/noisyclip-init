#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=/root/b1-env-20261027/bin/python
mkdir -p runs
exec 9>runs/pipeline.lock
flock -n 9 || exit 2
trap 'echo "FAILED line=$LINENO status=$?"; echo failed > runs/pipeline_status.txt' ERR
echo preparing > runs/pipeline_status.txt
git rev-parse HEAD > runs/git_commit.txt
nvidia-smi > runs/gpu.txt
"$PY" prepare_queue.py
"$PY" test_queue.py
if [[ ! -f runs/smoke_passed.json ]]; then
  echo smoke > runs/pipeline_status.txt
  "$PY" engine/train_stage1.py --workspace "$PWD" --config "$PWD/stage1.json" --run smoke --until 2
  "$PY" engine/train_stage1.py --workspace "$PWD" --config "$PWD/stage1.json" --run smoke --until 3 --resume
  "$PY" - <<'PY'
import json,torch
from pathlib import Path
c=torch.load('runs/smoke/last.pt',map_location='cpu',weights_only=False)
assert c['step']==3 and c['optimizer']['state'] and c['rng'][0]['stream_states']
assert c['contrast_queue']['version']==1 and c['contrast_queue']['features'].shape==(500,8,512)
assert all(torch.isfinite(v).all() for v in c['model'].values() if v.is_floating_point())
Path('runs/smoke_passed.json').write_text(json.dumps(dict(step=3,finite=True,resume_loaded=True,queue_saved=True,queue_warmup=128)))
PY
fi
echo training_stage1 > runs/pipeline_status.txt
args=(); [[ ! -f runs/stage1/last.pt ]] || args=(--resume)
"$PY" engine/train_stage1.py --workspace "$PWD" --config "$PWD/stage1.json" --run stage1 --until 4096 "${args[@]}"
echo validation_stage1 > runs/pipeline_status.txt
"$PY" evaluate.py --stage stage1
"$PY" compare.py
echo complete > runs/pipeline_status.txt
