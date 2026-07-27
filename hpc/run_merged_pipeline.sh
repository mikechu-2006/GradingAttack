#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH -t 1-00:00:00
#SBATCH -J merged_pipe
#SBATCH -o /dev/null
#SBATCH -e /dev/null

PROJECT_DIR="${PROJECT_DIR:-$HOME/GradingAttack}"
LOG_DIR="$PROJECT_DIR/hpc/logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/merged_pipeline_${SLURM_JOB_ID:-manual}.out" 2> "$LOG_DIR/merged_pipeline_${SLURM_JOB_ID:-manual}.err"

set -eo pipefail

ENV_NAME="${ENV_NAME:-gradingattack}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"
CONFIG_PATH="${CONFIG_PATH:-configs/Pipeline-GCG-SuffixBank-Llama-3.1-8B-Instruct-3c-heldout500-attention.yaml}"

cd "$PROJECT_DIR"

echo "[job] start $(date)"
echo "[job] host $(hostname)"
echo "[job] pwd $(pwd)"
echo "[job] id ${SLURM_JOB_ID:-manual}"
echo "[job] config $CONFIG_PATH"

module load anaconda3
module load "$CUDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -c "import torch, yaml, pandas; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'); print('pandas', pandas.__version__)"
python main.py "$CONFIG_PATH" --pipeline

echo "[job] done $(date)"