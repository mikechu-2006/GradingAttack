#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_grid_hs_%j.txt
#SBATCH -e err_grid_hs_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# 兼容入口 → run_pipeline_hijacking_suppression.sh（调参 + 全量）
echo "[sbatch-hs] job=${SLURM_JOB_ID:-unknown} (compat → hs-pipeline) $(date -Iseconds)"
exec bash run_pipeline_hijacking_suppression.sh
