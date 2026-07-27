#!/usr/bin/env bash
set -euo pipefail

cd /home/ruijia/wxz/GradingAttack-gh

CONDA=${CONDA:-/home/ruijia/miniconda3/bin/conda}
TMP_DIR=${TMP_DIR:-/tmp/grading_attack_qwen_injection_hs}

for shard in 0 1 2; do
  gpu=$((shard + 2))
  PYTHONPATH=/home/ruijia/wxz/GradingAttack-gh GRADING_ATTACK_DEVICE="cuda:$gpu" "$CONDA" run -n moe311 \
    python scripts/run_qwen_injection_hs_remaining.py \
      "$TMP_DIR/qwen3_4b_injection_hs.yaml" qwen3 "$shard" 3 &
done

for shard in 0 1 2; do
  gpu=$((shard + 5))
  PYTHONPATH=/home/ruijia/wxz/GradingAttack-gh GRADING_ATTACK_DEVICE="cuda:$gpu" "$CONDA" run -n moe311 \
    python scripts/run_qwen_injection_hs_remaining.py \
      "$TMP_DIR/qwen35_4b_injection_hs.yaml" qwen35 "$shard" 3 &
done

wait
