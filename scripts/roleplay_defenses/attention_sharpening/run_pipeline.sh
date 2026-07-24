#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-${REPO_ROOT}}"
source "${SCRIPT_DIR}/../run_env.sh"
PHASE="${AS_PIPELINE_PHASE:-all}"
echo "[roleplay-as] AS_PIPELINE_PHASE=${PHASE}"
exec "${PYTHON}" "${SCRIPT_DIR}/../attention_sharpening_pipeline.py" --phase "${PHASE}"
