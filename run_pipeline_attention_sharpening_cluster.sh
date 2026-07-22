#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# 工作目录 = sbatch 提交时所在目录（请在 ~/GradingAttack 下执行 sbatch）
echo "[sbatch] job=${SLURM_JOB_ID:-unknown} submit_dir=${SLURM_SUBMIT_DIR:-unknown} $(date -Iseconds)"

bash scripts/run_attention_sharpening_cluster.sh
