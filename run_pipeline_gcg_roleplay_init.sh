#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00


python main.py --pipeline configs/GCG-Llama-3.1-8B-Instruct-roleplay-init-cluster-2c.yaml
