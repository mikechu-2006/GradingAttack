#!/bin/bash
# 交互式作业 — Attention Sharpening smoke test
# 进入计算节点后执行:
#   conda activate gradingattack
#   bash scripts/run_attention_sharpening_smoke.sh

srun -p emergency_gpua40 \
     -n 8 \
     --gres=gpu:1 \
     --time=01:00:00 \
     --pty bash
