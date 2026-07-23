#!/bin/bash
# Hijacking Suppression smoke 测试（5 条）
set -euo pipefail

cd ~/GradingAttack

if [ "${CONDA_DEFAULT_ENV:-}" = "gradingattack" ] && [ -x "${CONDA_PREFIX:-}/bin/python" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
    echo "[smoke-hs] Using active conda env: gradingattack (${PYTHON})"
else
    # shellcheck disable=SC1091
    source scripts/activate_gradingattack_env.sh
    PYTHON=python
fi

echo "[smoke-hs] Running Hijacking Suppression pipeline (5 samples)..."
"${PYTHON}" main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-hijacking-suppression-smoke.yaml

echo
echo "[smoke-hs] Latest metrics:"
ls -lt result/RolePlay/Llama-3.1-8B-Instruct/*hijacking-suppression-smoke*_metrics.json | head -1
