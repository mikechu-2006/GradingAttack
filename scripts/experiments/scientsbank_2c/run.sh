#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-${REPO_ROOT}}"
source "${REPO_ROOT}/scripts/roleplay_defenses/run_env.sh"
PIPELINE="${PIPELINE:-all}"
PHASE="${PHASE:-all}"
SUFFIX_BANK_PATH="${SUFFIX_BANK_PATH:-}"
echo "[sb2c] PIPELINE=${PIPELINE} PHASE=${PHASE} SUFFIX_BANK_PATH=${SUFFIX_BANK_PATH:-<default>}"
export SUFFIX_BANK_PATH
exec "${PYTHON}" "${SCRIPT_DIR}/grid_search.py" --pipeline "${PIPELINE}" --phase "${PHASE}"
