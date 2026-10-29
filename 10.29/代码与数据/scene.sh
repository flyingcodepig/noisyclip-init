#!/usr/bin/env bash
set -euo pipefail
scene_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$scene_root"
scene_python="${PYTHON:-python}"
scene_action="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
case "$scene_action" in
  inspect) exec "$scene_python" -B checks/release.py "$@" ;;
  infer) exec "$scene_python" -B query/predict.py "$@" ;;
  cache) exec "$scene_python" -B ops/cache_train.py "$@" ;;
  init) exec "$scene_python" -B ops/fetch_weights.py "$@" ;;
  learn) exec "$scene_python" -B ops/train_chain.py "$@" ;;
  help|-h|--help) printf '%s\n' 'bash scene.sh inspect [--prediction CSV]' 'bash scene.sh infer [--test-dir DIR --output NEW_DIR]' 'bash scene.sh cache [--train-dir DIR --cache DIR]' 'bash scene.sh init [--destination DIR]' 'bash scene.sh learn [--dry-run | --stage all|initial|alignment|refine]' ;;
  *) printf 'Unknown scene action: %s\n' "$scene_action" >&2; exit 2 ;;
esac
