set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$HERE/runs/stage1/last.pt}"
[[ -f "$STAGE1_CHECKPOINT" ]] || { echo "Stage-1 checkpoint not found: $STAGE1_CHECKPOINT" >&2; exit 2; }
STAGE1_CHECKPOINT="$(realpath "$STAGE1_CHECKPOINT")"
[[ -d "$HERE/data/official/train" ]] || { echo "Official TRAIN images not found: $HERE/data/official/train" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
(( GPU_COUNT >= NPROC_PER_NODE )) || { echo "At least $NPROC_PER_NODE visible CUDA GPUs are required" >&2; exit 2; }
cd "$HERE"
"$PYTHON" fetch_weights.py
"$PYTHON" build_cache.py --split train
"$PYTHON" verify_candidates.py --train-root "$HERE/data/official/train"
[[ -f "$HERE/cache/train_quality.npz" ]] || { echo "Run train_stage1.sh to build B1 quality inputs first" >&2; exit 2; }
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" engine/train_with_cutmix_four_gpu.py --workspace "$HERE" --config "$HERE/train_config.json" --run stage2 --until 1536 --restart-from "$STAGE1_CHECKPOINT"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" engine/train_with_cutmix_four_gpu.py --workspace "$HERE" --config "$HERE/train_config.json" --run stage2 --until 2048 --resume
