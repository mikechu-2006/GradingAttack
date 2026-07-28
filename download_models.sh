#!/bin/bash
#SBATCH -p cpu
#SBATCH -o download_models_%j.txt
#SBATCH -e download_models_%j.txt
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --mem=16G

# Activate conda environment (needed for modelscope / huggingface_hub)
source scripts/activate_gradingattack_env.sh

# Use ModelScope to fetch models
export VLLM_USE_MODELSCOPE=True

echo "Pre-downloading models from ModelScope..."
echo "----------------------------------------"

python download_models.py

echo "----------------------------------------"
echo "Done. Check the output above for per-model status."
