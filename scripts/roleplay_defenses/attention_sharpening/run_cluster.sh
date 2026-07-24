#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-${REPO_ROOT}}"
source "${SCRIPT_DIR}/../run_env.sh"
CONFIG="configs/roleplay/attention_sharpening/best-2c.yaml"
FALLBACK="configs/roleplay/attention_sharpening/cluster-2c.yaml"
if [ -f "${CONFIG}" ]; then
    echo "[roleplay-as] using best config: ${CONFIG}"
else
    CONFIG="${FALLBACK}"
    echo "[roleplay-as] using cluster config: ${CONFIG}"
fi
exec "${PYTHON}" main.py --pipeline "${CONFIG}"
