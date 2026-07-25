#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# Delimiter Confusion (DC) — XML structure exploit
python main.py --pipeline configs/injection/dc-cluster-2c.yaml
