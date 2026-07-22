#!/bin/bash
# Attention Sharpening 全量 cluster 管线（供 sbatch 调用）
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${HOME}/GradingAttack}"

# shellcheck disable=SC1091
source scripts/activate_gradingattack_env.sh

echo "[cluster] python=$(command -v python)"
echo "[cluster] config=RolePlay-Llama-3.1-8B-Instruct-attention-sharpening-cluster.yaml"

exec python main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-attention-sharpening-cluster.yaml
