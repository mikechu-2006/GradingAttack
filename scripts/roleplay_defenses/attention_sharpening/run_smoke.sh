#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-${REPO_ROOT}}"
source "${SCRIPT_DIR}/../run_env.sh"
CONFIG="${1:-configs/roleplay/attention_sharpening/smoke-2c.yaml}"
echo "[roleplay-as] config=${CONFIG}"
exec "${PYTHON}" main.py --pipeline "${CONFIG}"
