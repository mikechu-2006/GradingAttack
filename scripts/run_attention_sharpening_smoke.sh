#!/bin/bash
set -euo pipefail

cd ~/GradingAttack

echo "[smoke] Running Attention Sharpening pipeline (5 samples)..."
python main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-attention-sharpening-smoke.yaml

echo
echo "[smoke] Latest metrics:"
ls -lt result/RolePlay/Llama-3.1-8B-Instruct/*attention-sharpening-smoke*_metrics.json | head -1
echo
cat result/RolePlay/Llama-3.1-8B-Instruct/*attention-sharpening-smoke*_metrics.json
