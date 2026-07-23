#!/bin/bash
# Hijacking Suppression 网格搜索 + 全量最优参数评估（供 sbatch 调用）
set -eo pipefail

echo "[grid-hs] start time=$(date -Iseconds) host=$(hostname)"

cd "${SLURM_SUBMIT_DIR:-${HOME}/GradingAttack}" || {
    echo "[grid-hs] FATAL: cannot cd to GradingAttack" >&2
    exit 1
}
echo "[grid-hs] pwd=$(pwd)"

PYTHON=""

if source scripts/activate_gradingattack_env.sh 2>&1; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON="${CONDA_PREFIX}/bin/python"
    fi
fi

if [ -z "${PYTHON}" ]; then
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
    echo "[grid-hs] FATAL: no usable python found" >&2
    exit 1
fi

echo "[grid-hs] python=${PYTHON}"
echo "[grid-hs] version=$("${PYTHON}" --version 2>&1)"

PHASE="${GRID_SEARCH_PHASE:-all}"
echo "[grid-hs] GRID_SEARCH_PHASE=${PHASE}"

exec "${PYTHON}" scripts/grid_search_hijacking_suppression.py --phase "${PHASE}"
