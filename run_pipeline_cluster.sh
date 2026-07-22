#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D ~/GradingAttack
#SBATCH -t 7-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gradingattack

python main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-cluster.yaml
