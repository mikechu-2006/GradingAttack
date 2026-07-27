#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o demo_output_%j.txt
#SBATCH -e demo_err_%j.txt
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -t 17:00:00

# Activate conda environment
source scripts/activate_gradingattack_env.sh

# Set your API keys here (or export them before submitting)
# export DEEPSEEK_API_KEY="your-key-here"
# export LOCAL_LLM_API_KEY="your-key-here"

echo "Launching GradingAttack Demo..."
echo "Check the output below for the gradio.live share link:"
echo "----------------------------------------"

python demo_app.py
