#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# Instruction Mimicry (IM) + Hijacking Suppression — SciEntsBank 2c tune 5 / full 50
export PIPELINE=im_hs
export PHASE=all
bash scripts/experiments/scientsbank_2c/run.sh
