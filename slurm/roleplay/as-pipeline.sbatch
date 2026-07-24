#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o logs/slurm/roleplay-as-pipeline_%j.out
#SBATCH -e logs/slurm/roleplay-as-pipeline_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

mkdir -p logs/slurm
bash scripts/roleplay_defenses/attention_sharpening/run_pipeline.sh
