#!/bin/bash
# 统一 grid search（AS / HS / all）
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-${REPO_ROOT}}"
source "${SCRIPT_DIR}/run_env.sh"
DEFENSE_TYPE="${DEFENSE_TYPE:-all}"
PHASE="${GRID_SEARCH_PHASE:-all}"
echo "[grid] DEFENSE_TYPE=${DEFENSE_TYPE} GRID_SEARCH_PHASE=${PHASE}"
exec "${PYTHON}" "${SCRIPT_DIR}/grid_search.py" --defense "${DEFENSE_TYPE}" --phase "${PHASE}"
