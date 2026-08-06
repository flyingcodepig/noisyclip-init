#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 PHYSICAL_GPU_ID CONFIG1 [CONFIG2 ...]" >&2
  exit 2
fi

gpu="$1"
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for config in "$@"; do
  echo "Queue item: ${config}"
  bash "${script_dir}/run_experiment.sh" "${config}" "${gpu}"
done

