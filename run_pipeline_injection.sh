#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# Default: Delimiter Confusion (DC) strategy — XML structure exploit
python main.py --pipeline configs/injection/dc-cluster-2c.yaml

# Alternative strategies — uncomment to run instead:
# python main.py --pipeline configs/injection/ao-cluster-2c.yaml
# python main.py --pipeline configs/injection/im-cluster-2c.yaml
