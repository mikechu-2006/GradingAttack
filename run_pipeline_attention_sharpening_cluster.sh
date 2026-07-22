#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D ~/GradingAttack
#SBATCH -t 7-00:00:00

bash scripts/run_attention_sharpening_cluster.sh
