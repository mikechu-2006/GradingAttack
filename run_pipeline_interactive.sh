#!/bin/bash
# 交互式作业 — 申请计算节点并进入交互 bash
# 进入后手动执行: conda activate gradingattack && python main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-cluster.yaml

slurm -p emergency_gpua40 \
     -n 8 \
     --gres=gpu:1 \
     --time=01:00:00 \
     --pty bash
