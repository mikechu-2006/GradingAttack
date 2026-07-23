#!/bin/bash
# RolePlay 注意力防御统一网格搜索（供 sbatch 调用）
set -eo pipefail

echo "[grid] start time=$(date -Iseconds) host=$(hostname)"

cd "${SLURM_SUBMIT_DIR:-${HOME}/GradingAttack}" || {
    echo "[grid] FATAL: cannot cd to GradingAttack" >&2
    exit 1
}
echo "[grid] pwd=$(pwd)"

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
    echo "[grid] FATAL: no usable python found" >&2
    exit 1
fi

echo "[grid] python=${PYTHON}"
echo "[grid] version=$("${PYTHON}" --version 2>&1)"

# DEFENSE_TYPE: attention_sharpening | hijacking_suppression | all
DEFENSE_TYPE="${DEFENSE_TYPE:-all}"
PHASE="${GRID_SEARCH_PHASE:-all}"
echo "[grid] DEFENSE_TYPE=${DEFENSE_TYPE} GRID_SEARCH_PHASE=${PHASE}"

exec "${PYTHON}" scripts/grid_search_roleplay_defense.py \
    --defense "${DEFENSE_TYPE}" \
    --phase "${PHASE}"
