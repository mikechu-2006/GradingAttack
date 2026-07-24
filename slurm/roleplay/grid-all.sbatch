#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o logs/slurm/roleplay-grid_%j.out
#SBATCH -e logs/slurm/roleplay-grid_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# DEFENSE_TYPE=all|attention_sharpening|hijacking_suppression
# GRID_SEARCH_PHASE=all|tune|full
mkdir -p logs/slurm
bash scripts/roleplay_defenses/run_grid_search.sh
