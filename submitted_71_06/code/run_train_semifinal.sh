#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export PATH="/root/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p logs runs
exec 9>runs/pipeline.lock
flock -n 9 || { echo 'Training pipeline is already active'; exit 2; }
echo "$$" > runs/pipeline.pid
status() {
  python -c 'import json,sys,time; from pathlib import Path; p=Path("runs/pipeline_status.json"); q=p.with_suffix(".tmp"); q.write_text(json.dumps(dict(stage=sys.argv[1],unix_time=time.time(),pid=int(Path("runs/pipeline.pid").read_text())),indent=2)); q.replace(p)' "$1"
}
trap 'code=$?; status failed; printf "Pipeline failed with exit code %s at line %s\n" "$code" "$LINENO"; exit "$code"' ERR
status waiting_for_verified_train_upload
until [[ -f TRAIN_UPLOAD_VERIFIED ]]; do sleep 10; done
status verifying_preflight
until [[ -f PREFLIGHT_VERIFIED ]]; do
  [[ ! -f PREFLIGHT_FAILED ]] || { echo 'Preflight failed; inspect logs_preflight.log'; status failed; exit 1; }
  sleep 10
done
status verifying_official_initialization
if [[ ! -f weights/clip-vit-base-patch32/pytorch_model.bin ]]; then
  until [[ -f weights/clip-vit-base-patch32/pytorch_model.bin.downloading ]] && [[ "$(stat -c %s weights/clip-vit-base-patch32/pytorch_model.bin.downloading)" == 605247071 ]]; do sleep 10; done
  echo 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f  weights/clip-vit-base-patch32/pytorch_model.bin.downloading' | sha256sum -c -
  mv weights/clip-vit-base-patch32/pytorch_model.bin.downloading weights/clip-vit-base-patch32/pytorch_model.bin
fi
python fetch_weights.py
status checking_gpu_memory
if [[ ! -f hardware_profile.json ]]; then
  for batch in 48 24 12; do
    if python smoke_gpu.py "$batch" > "logs/smoke_${batch}.log" 2>&1; then
      printf '{"microbatch":%s}\n' "$batch" > hardware_profile.json
      break
    fi
    grep -q 'out of memory' "logs/smoke_${batch}.log" || { cat "logs/smoke_${batch}.log"; exit 1; }
  done
fi
test -f hardware_profile.json
status preparing_semifinal_train_features
python prepare_semifinal.py
status building_semifinal_train_caches
python build_semifinal_cache.py
python configure_semifinal.py
status training_stage1
resume=()
[[ ! -f runs/stage1/last.pt ]] || resume=(--resume)
python engine/train_stage1.py --workspace "$PWD" --config "$PWD/semifinal_stage1.json" --run stage1 --until 4096 "${resume[@]}"
status training_stage2
if [[ -f runs/stage2/last.pt ]]; then
  python engine/train_with_cutmix_four_gpu.py --workspace "$PWD" --config "$PWD/semifinal_stage2.json" --run stage2 --until 1536 --resume
else
  python engine/train_with_cutmix_four_gpu.py --workspace "$PWD" --config "$PWD/semifinal_stage2.json" --run stage2 --until 1536 --restart-from "$PWD/runs/stage1/last.pt"
fi
python engine/train_with_cutmix_four_gpu.py --workspace "$PWD" --config "$PWD/semifinal_stage2.json" --run stage2 --until 2048 --resume
python - <<'PY'
import json, torch
from pathlib import Path
from fetch_weights import digest
p=Path('runs/stage2/model.pt')
c=torch.load(p,map_location='cpu',weights_only=False)
assert c['step']==2048 and c['model']['head.weight'].shape==(500,512)
Path('runs/final_model_manifest.json').write_text(json.dumps(dict(model=str(p),sha256=digest(p),step=c['step'],classes=500,training_data='current semifinal TRAIN only'),indent=2))
print('Semifinal training completed: runs/stage2/model.pt',flush=True)
PY
status completed
