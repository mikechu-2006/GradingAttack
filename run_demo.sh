#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o demo_output_%j.txt
#SBATCH -e demo_err_%j.txt
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -t 17:00:00

# Activate conda environment
source scripts/activate_gradingattack_env.sh

# For DeepSeek API (remote model only)
# export DEEPSEEK_API_KEY="your-key-here"

# Use ModelScope to download / load models
export VLLM_USE_MODELSCOPE=True

echo "Launching GradingAttack Demo..."
echo "Local models run via vLLM; deepseek-chat uses API."
echo "Check below for the gradio.live share link:"
echo "----------------------------------------"

python demo_app.py
