#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_hs_%j.txt
#SBATCH -e err_hs_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# Hijacking Suppression 单次全量。请在 ~/GradingAttack 下 sbatch。
echo "[sbatch-hs] job=${SLURM_JOB_ID:-unknown} submit_dir=${SLURM_SUBMIT_DIR:-unknown} $(date -Iseconds)"

bash scripts/run_hijacking_suppression_cluster.sh
