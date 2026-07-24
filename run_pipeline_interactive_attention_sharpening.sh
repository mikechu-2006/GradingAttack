#!/bin/bash
# 交互式作业 — Attention Sharpening 全量/调试
# 进入计算节点后执行:
#   module load anaconda3 && eval "$(conda shell.bash hook)" && conda activate gradingattack
#   python main.py --pipeline configs/roleplay/attention_sharpening/cluster-2c.yaml
#
# 或提交批处理（推荐）:
#   sbatch slurm/roleplay/as-cluster.sbatch

slurm -p emergency_gpua40 \
     -n 8 \
     --gres=gpu:1 \
     --time=07:00:00 \
     --pty bash