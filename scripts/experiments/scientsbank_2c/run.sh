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
case "${PIPELINE}" in
  gcg_hs|gcg_as|all)
    if [ ! -f "${SUFFIX_BANK_PATH}" ]; then
      echo "[sb2c] FATAL: suffix bank not found: ${SUFFIX_BANK_PATH}" >&2
      echo "[sb2c] Place gcg_suffix_bank_2c_his.jsonl under result_from_hpc_gcg_suffix_bank/," >&2
      echo "[sb2c] or build a fresh bank via scripts/experiments/scientsbank_2c/build_suffix_bank.sh" >&2
      exit 1
    fi
    ;;
esac
SB2C_ARGS=()
if [ "${REBUILD_SUMMARY:-0}" = "1" ]; then
  SB2C_ARGS+=(--rebuild-summary)
fi
if [ "${FORCE_RERUN:-0}" = "1" ]; then
  SB2C_ARGS+=(--force-rerun)
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  SB2C_ARGS+=(--dry-run)
fi
# GCG / Injection 管线：HPC 上 tune 常已完成，默认允许从已有 metrics 进入 full
case "${PIPELINE}" in
  gcg_hs|gcg_as|ao_hs|ao_as|dc_hs|dc_as|im_hs|im_as)
    ALLOW_PARTIAL_TUNE="${ALLOW_PARTIAL_TUNE:-1}"
    ;;
esac
if [ "${ALLOW_PARTIAL_TUNE:-0}" = "1" ]; then
  SB2C_ARGS+=(--allow-partial-tune)
fi
exec "${PYTHON}" "${SCRIPT_DIR}/grid_search.py" --pipeline "${PIPELINE}" --phase "${PHASE}" "${SB2C_ARGS[@]}"
