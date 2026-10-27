#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$HERE/../code"
PYTHON="${PYTHON:-/root/b1-env-20261027/bin/python}"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$CODE/cache/a100"
exec 9>"$CODE/cache/a100/.prepare.lock"
flock -n 9 || exit 2
"$PYTHON" "$HERE/fetch_official.py"
[[ -f "$CODE/data/a100/source_manifest.json" ]] || { echo 'Data verification has not completed'; exit 2; }
"$PYTHON" "$HERE/prepare_features.py"
"$PYTHON" "$HERE/build_actual_quality.py"
mkdir -p /root/b1-image-cache-20261027
if [[ ! -e "$CODE/image_cache" ]]; then
  ln -s /root/b1-image-cache-20261027 "$CODE/image_cache"
fi
"$PYTHON" "$CODE/build_cache_a100.py" --split train --workers 12
