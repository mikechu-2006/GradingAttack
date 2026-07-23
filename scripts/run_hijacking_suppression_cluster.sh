#!/bin/bash
# Hijacking Suppression 全量 cluster 管线（供 sbatch 调用）
set -eo pipefail

echo "[cluster-hs] start time=$(date -Iseconds) host=$(hostname)"

cd "${SLURM_SUBMIT_DIR:-${HOME}/GradingAttack}" || {
    echo "[cluster-hs] FATAL: cannot cd to GradingAttack" >&2
    exit 1
}
echo "[cluster-hs] pwd=$(pwd)"

PYTHON=""

if source scripts/activate_gradingattack_env.sh 2>&1; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON="${CONDA_PREFIX}/bin/python"
    fi
fi

if [ -z "${PYTHON}" ]; then
    echo "[cluster-hs] activate script path failed, trying fallbacks..."
    if [ -f /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
        conda activate gradingattack 2>/dev/null || true
        if [ -x "${CONDA_PREFIX:-}/bin/python" ]; then
            PYTHON="${CONDA_PREFIX}/bin/python"
        fi
    fi
fi

if [ -z "${PYTHON}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    fi
fi

if [ -z "${PYTHON}" ] || [ ! -x "${PYTHON}" ]; then
    echo "[cluster-hs] FATAL: no usable python found" >&2
    exit 1
fi

echo "[cluster-hs] python=${PYTHON}"
echo "[cluster-hs] version=$("${PYTHON}" --version 2>&1)"
echo "[cluster-hs] config=RolePlay-Llama-3.1-8B-Instruct-hijacking-suppression-cluster.yaml"

exec "${PYTHON}" main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-hijacking-suppression-cluster.yaml
