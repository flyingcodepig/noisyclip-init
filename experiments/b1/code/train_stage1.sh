set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
[[ -d "$HERE/data/official/train" ]] || { echo "Official TRAIN images not found: $HERE/data/official/train" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
(( GPU_COUNT >= NPROC_PER_NODE )) || { echo "At least $NPROC_PER_NODE visible CUDA GPUs are required" >&2; exit 2; }
cd "$HERE"
"$PYTHON" fetch_weights.py
"$PYTHON" build_cache.py --split train
"$PYTHON" generate_train_candidates.py --train-root "$HERE/data/official/train" --output "$HERE/cache/train_candidate_labels.npy" --metadata "$HERE/cache/train_candidate_labels.json"
"$PYTHON" verify_candidates.py --train-root "$HERE/data/official/train"
"$PYTHON" prepare_training.py
"$PYTHON" build_quality.py
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" engine/train_stage1.py --workspace "$HERE" --config "$HERE/train_config_stage1.json" --run stage1 --until 4096
