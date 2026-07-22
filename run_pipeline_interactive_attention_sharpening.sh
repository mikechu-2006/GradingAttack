#!/bin/bash
# 交互式作业 — Attention Sharpening smoke test
# 进入计算节点后执行:
#   bash scripts/run_attention_sharpening_smoke.sh
#
# 若 conda 仍找不到，先在登录节点执行 which conda，再编辑
# scripts/activate_gradingattack_env.sh 补充正确路径。

srun -p emergency_gpua40 \
     -n 8 \
     --gres=gpu:1 \
     --time=01:00:00 \
     --pty bash