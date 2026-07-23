#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH -t 00:10:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null

PROJECT_DIR="${PROJECT_DIR:-$HOME/GradingAttack}"
LOG_DIR="$PROJECT_DIR/hpc/logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/test_gpu_${SLURM_JOB_ID:-manual}.out" 2> "$LOG_DIR/test_gpu_${SLURM_JOB_ID:-manual}.err"

set -eo pipefail

ENV_NAME="${ENV_NAME:-gradingattack}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"

cd "$PROJECT_DIR"
module load anaconda3
module load "$CUDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[test_gpu] Date: $(date)"
echo "[test_gpu] Host: $(hostname)"
echo "[test_gpu] Workdir: $(pwd)"
echo "[test_gpu] Job ID: ${SLURM_JOB_ID:-manual}"
echo "[test_gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "[test_gpu] nvidia-smi:"
nvidia-smi

echo "[test_gpu] torch check:"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('cuda version', torch.version.cuda)
print('device count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
