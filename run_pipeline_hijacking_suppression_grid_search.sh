#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_grid_hs_%j.txt
#SBATCH -e err_grid_hs_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# 兼容入口 → 统一脚本（仅 Hijacking Suppression）
export DEFENSE_TYPE=hijacking_suppression
exec bash run_pipeline_roleplay_defense_grid_search.sh
