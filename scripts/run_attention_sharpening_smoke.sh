#!/bin/bash
set -euo pipefail

cd ~/GradingAttack

if [ "${CONDA_DEFAULT_ENV:-}" = "gradingattack" ] && [ -x "${CONDA_PREFIX:-}/bin/python" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
    echo "[smoke] Using active conda env: gradingattack (${PYTHON})"
else
    # shellcheck disable=SC1091
    source scripts/activate_gradingattack_env.sh
    PYTHON=python
fi

echo "[smoke] Running Attention Sharpening pipeline (5 samples)..."
"${PYTHON}" main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-attention-sharpening-smoke.yaml

echo
echo "[smoke] Latest metrics:"
ls -lt result/RolePlay/Llama-3.1-8B-Instruct/*attention-sharpening-smoke*_metrics.json | head -1
echo
cat result/RolePlay/Llama-3.1-8B-Instruct/*attention-sharpening-smoke*_metrics.json
