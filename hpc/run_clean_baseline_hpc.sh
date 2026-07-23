#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH -t 1-00:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null

PROJECT_DIR="${PROJECT_DIR:-$HOME/GradingAttack}"
LOG_DIR="$PROJECT_DIR/hpc/logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/clean_baseline_${SLURM_JOB_ID:-manual}.out" 2> "$LOG_DIR/clean_baseline_${SLURM_JOB_ID:-manual}.err"

set -eo pipefail

ENV_NAME="${ENV_NAME:-gradingattack}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"
CONFIG_PATH="${CONFIG_PATH:-configs/GCG-Llama-3.1-8B-Instruct-defense.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
OUTPUT_PATH="${OUTPUT_PATH:-result_baseline/scientsbank_clean_baseline.jsonl}"

cd "$PROJECT_DIR"
mkdir -p result_baseline
module load anaconda3
module load "$CUDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[baseline] Date: $(date)"
echo "[baseline] Host: $(hostname)"
echo "[baseline] Workdir: $(pwd)"
echo "[baseline] Job ID: ${SLURM_JOB_ID:-manual}"
echo "[baseline] Config: $CONFIG_PATH"
echo "[baseline] Max samples: $MAX_SAMPLES"
echo "[baseline] Output: $OUTPUT_PATH"
echo "[baseline] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python baseline_eval.py \
  --config "$CONFIG_PATH" \
  --max_samples "$MAX_SAMPLES" \
  --output "$OUTPUT_PATH"

echo "[baseline] Done at $(date)"
