#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: bash $0 --env PATH [--build|--check|--shell]"
}

env_file=""
action=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) env_file="${2:-}"; shift 2 ;;
    --build|--check|--shell) action="$1"; shift ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "${env_file}" || -z "${action}" || ! -f "${env_file}" ]]; then
  usage
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

required=(PROJECT_ROOT TRAIN_ROOT TEST_ROOT RUN_ROOT CACHE_ROOT)
for name in "${required[@]}"; do
  value="${!name:-}"
  if [[ -z "${value}" || "${value}" != /* ]]; then
    echo "${name} must be an absolute path." >&2
    exit 3
  fi
done

if [[ "${TRAIN_ROOT}" == "${TEST_ROOT}" || "${TRAIN_ROOT}" == "${RUN_ROOT}" || "${TEST_ROOT}" == "${RUN_ROOT}" ]]; then
  echo "Train, test, and run roots must be distinct." >&2
  exit 3
fi

for path in "${PROJECT_ROOT}" "${TRAIN_ROOT}" "${TEST_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Required directory does not exist: ${path}" >&2
    exit 3
  fi
done
mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}"

compose_file="${PROJECT_ROOT}/init_build/02_deployment/compose.yaml"
if [[ ! -f "${compose_file}" ]]; then
  echo "Compose file not found: ${compose_file}" >&2
  exit 3
fi

case "${action}" in
  --build)
    docker compose --env-file "${env_file}" -f "${compose_file}" build --pull
    ;;
  --check)
    docker compose --env-file "${env_file}" -f "${compose_file}" run --rm train \
      python init_build/02_deployment/healthcheck.py
    ;;
  --shell)
    docker compose --env-file "${env_file}" -f "${compose_file}" run --rm train bash
    ;;
esac

