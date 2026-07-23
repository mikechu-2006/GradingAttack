#!/bin/bash
# Hijacking Suppression 管线：小样本调参 → 最优参数全量（供 sbatch 调用）
set -eo pipefail

echo "[hs-pipeline] start time=$(date -Iseconds) host=$(hostname)"

cd "${SLURM_SUBMIT_DIR:-${HOME}/GradingAttack}" || {
    echo "[hs-pipeline] FATAL: cannot cd to GradingAttack" >&2
    exit 1
}
echo "[hs-pipeline] pwd=$(pwd)"

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
    echo "[hs-pipeline] FATAL: no usable python found" >&2
    exit 1
fi

echo "[hs-pipeline] python=${PYTHON}"
echo "[hs-pipeline] version=$("${PYTHON}" --version 2>&1)"

# HS_PIPELINE_PHASE: tune | full | all  (默认 all = 调参 + 全量)
PHASE="${HS_PIPELINE_PHASE:-all}"
echo "[hs-pipeline] HS_PIPELINE_PHASE=${PHASE}"

exec "${PYTHON}" scripts/hijacking_suppression_pipeline.py --phase "${PHASE}"
