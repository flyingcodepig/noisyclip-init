#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash $0 CONFIG_YAML PHYSICAL_GPU_ID" >&2
  exit 2
fi

config="$(realpath "$1")"
physical_gpu="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  echo "GPU ID must be a non-negative integer." >&2
  exit 2
fi

run_id="$(python "${script_dir}/make_run_id.py" "${config}")"
export NOISYCLIP_RUN_ID="${run_id}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"

lock_file="/tmp/noisyclip-gpu-${physical_gpu}.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "GPU ${physical_gpu} is already reserved by another noisyclip run." >&2
  exit 6
fi

python "${script_dir}/validate_config.py" --config "${config}" --require-env
python "${script_dir}/preflight.py" --config "${config}"

echo "Starting run ${run_id} on physical GPU ${physical_gpu}."
python -m noisyclip.cli.train \
  --config "${config}" \
  --run-id "${run_id}"
