#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-${REPO_ROOT}}"
source "${REPO_ROOT}/scripts/roleplay_defenses/run_env.sh"
PIPELINE="${PIPELINE:-all}"
PHASE="${PHASE:-all}"
DEFAULT_SUFFIX_BANK="result_from_hpc_gcg_suffix_bank/gcg_suffix_bank_2c_his.jsonl"
SUFFIX_BANK_PATH="${SUFFIX_BANK_PATH:-${DEFAULT_SUFFIX_BANK}}"
echo "[sb2c] PIPELINE=${PIPELINE} PHASE=${PHASE} SUFFIX_BANK_PATH=${SUFFIX_BANK_PATH}"
export SUFFIX_BANK_PATH
SB2C_ARGS=()
if [ "${REBUILD_SUMMARY:-0}" = "1" ]; then
  SB2C_ARGS+=(--rebuild-summary)
fi
if [ "${ALLOW_PARTIAL_TUNE:-0}" = "1" ]; then
  SB2C_ARGS+=(--allow-partial-tune)
fi
if [ "${FORCE_RERUN:-0}" = "1" ]; then
  SB2C_ARGS+=(--force-rerun)
fi
exec "${PYTHON}" "${SCRIPT_DIR}/grid_search.py" --pipeline "${PIPELINE}" --phase "${PHASE}" "${SB2C_ARGS[@]}"
