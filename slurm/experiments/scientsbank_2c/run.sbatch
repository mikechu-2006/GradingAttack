#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o logs/slurm/scientsbank-2c_%j.out
#SBATCH -e logs/slurm/scientsbank-2c_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# SciEntsBank 2c 四管线: tune → full (100)
# 默认 suffix bank: result_from_hpc_gcg_suffix_bank/gcg_suffix_bank_2c_his.jsonl
# 示例:
#   PIPELINE=gcg_hs sbatch slurm/experiments/scientsbank_2c/run.sbatch
#   PIPELINE=all PHASE=tune sbatch slurm/experiments/scientsbank_2c/run.sbatch

mkdir -p logs/slurm
bash scripts/experiments/scientsbank_2c/run.sh
