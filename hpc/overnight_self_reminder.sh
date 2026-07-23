#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH -t 7-00:00:00
#SBATCH -J self_reminder
#SBATCH -o /dev/null
#SBATCH -e /dev/null

PROJECT_DIR="${PROJECT_DIR:-$HOME/GradingAttack}"
LOG_DIR="$PROJECT_DIR/hpc/logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/self_reminder_${SLURM_JOB_ID:-manual}.out" 2> "$LOG_DIR/self_reminder_${SLURM_JOB_ID:-manual}.err"

set -eo pipefail

cd "$PROJECT_DIR"

echo "[job] start $(date)"
echo "[job] host $(hostname)"
echo "[job] pwd $(pwd)"
echo "[job] id ${SLURM_JOB_ID:-manual}"

module load anaconda3
module load cuda/12.4
eval "$(conda shell.bash hook)"
conda activate gradingattack

# Avoid importing unused optional backends for this RolePlay pipeline run.
printf '' > baselines/__init__.py
printf '' > baselines/gcg/__init__.py
printf '' > baselines/roleplay/__init__.py

sed -i 's/debug: true/debug: false/' configs/RolePlay-Llama-3.1-8B-Instruct-cluster.yaml

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python main.py --pipeline configs/RolePlay-Llama-3.1-8B-Instruct-cluster.yaml

echo "[job] end $(date)"