#!/bin/bash
# Interactive demo launch — request a 12-hour GPU node via srun.
# Usage:  bash run_demo_interactive.sh
#
# Once the node is allocated you'll be dropped into a shell.
# Then run:
#   source scripts/activate_gradingattack_env.sh
#   export DEEPSEEK_API_KEY="your-key-here"   # optional, for deepseek-chat
#   python demo_app.py

srun -p emergency_gpua40 \
    --gres=gpu:1 \
    -n 4 \
    --time=08:00:00 \
    --pty bash
