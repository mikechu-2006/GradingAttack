#!/bin/bash
# Attention Sharpening 全量 cluster 管线（供 sbatch 调用）
set -eo pipefail

echo "[cluster] start time=$(date -Iseconds) host=$(hostname)"

cd "${SLURM_SUBMIT_DIR:-${HOME}/GradingAttack}" || {
    echo "[cluster] FATAL: cannot cd to GradingAttack" >&2
    exit 1
}
echo "[cluster] pwd=$(pwd)"

PYTHON=""

# 1) 尝试标准 activate 脚本
if source scripts/activate_gradingattack_env.sh 2>&1; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON="${CONDA_PREFIX}/bin/python"
    fi
fi

# 2) 兜底：系统 anaconda + gradingattack
if [ -z "${PYTHON}" ]; then
    echo "[cluster] activate script path failed, trying fallbacks..."
    if [ -f /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
        conda activate gradingattack 2>/dev/null || true
        if [ -x "${CONDA_PREFIX:-}/bin/python" ]; then
            PYTHON="${CONDA_PREFIX}/bin/python"
        fi
    fi
fi

# 3) 最后兜底：PATH 中的 python（与原 run_pipeline_cluster.sh 行为一致）
if [ -z "${PYTHON}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    fi
fi

if [ -z "${PYTHON}" ] || [ ! -x "${PYTHON}" ]; then
    echo "[cluster] FATAL: no usable python found" >&2
    echo "[cluster] PATH=${PATH}" >&2
    exit 1
fi

echo "[cluster] python=${PYTHON}"
echo "[cluster] version=$("${PYTHON}" --version 2>&1)"
echo "[cluster] config=RolePlay-Llama-3.1-8B-Instruct-attention-sharpening-cluster.yaml"

exec "${PYTHON}" main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-attention-sharpening-cluster.yaml
