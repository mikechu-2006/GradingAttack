#!/bin/bash
# 激活 HPC Python 环境，设置 PYTHON 变量
set -eo pipefail

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

if [ -z "${PYTHON}" ] && command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
fi

if [ -z "${PYTHON}" ] || [ ! -x "${PYTHON}" ]; then
    echo "[roleplay] FATAL: no usable python found" >&2
    exit 1
fi

export PYTHON
