#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export PATH="/root/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
mkdir -p logs
trap 'code=$?; if ((code != 0)); then touch PREFLIGHT_FAILED; fi' EXIT
if [[ ! -f weights/clip-vit-base-patch32/pytorch_model.bin ]]; then
  until [[ -f weights/clip-vit-base-patch32/pytorch_model.bin ]] || { [[ -f weights/clip-vit-base-patch32/pytorch_model.bin.downloading ]] && [[ "$(stat -c %s weights/clip-vit-base-patch32/pytorch_model.bin.downloading)" == 605247071 ]]; }; do sleep 10; done
  if [[ ! -f weights/clip-vit-base-patch32/pytorch_model.bin ]]; then
    echo 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f  weights/clip-vit-base-patch32/pytorch_model.bin.downloading' | sha256sum -c -
    mv weights/clip-vit-base-patch32/pytorch_model.bin.downloading weights/clip-vit-base-patch32/pytorch_model.bin
  fi
fi
for batch in 48 24 12; do
  if python smoke_gpu.py "$batch" > "logs/smoke_${batch}.log" 2>&1; then
    printf '{"microbatch":%s}\n' "$batch" > hardware_profile.json
    break
  fi
  grep -q 'out of memory' "logs/smoke_${batch}.log" || { cat "logs/smoke_${batch}.log"; exit 1; }
done
test -f hardware_profile.json
python check_training_resume.py
touch PREFLIGHT_VERIFIED
