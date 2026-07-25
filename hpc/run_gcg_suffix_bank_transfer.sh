#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH -t 1-00:00:00
#SBATCH -J gcg_bank
#SBATCH -o /dev/null
#SBATCH -e /dev/null

PROJECT_DIR="${PROJECT_DIR:-$HOME/GradingAttack}"
LOG_DIR="$PROJECT_DIR/hpc/logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/gcg_suffix_bank_${SLURM_JOB_ID:-manual}.out" 2> "$LOG_DIR/gcg_suffix_bank_${SLURM_JOB_ID:-manual}.err"

set -eo pipefail

ENV_NAME="${ENV_NAME:-gradingattack}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"
BANK_PATH="${BANK_PATH:-gcg_suffix_bank.txt}"
DATA_PATH="${DATA_PATH:-dataset/scientsbank.jsonl}"
TEMPLATE_PATH="${TEMPLATE_PATH:-configs/grading_template_ci_2c.txt}"
NCLASS="${NCLASS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
RANDOM_SEED="${RANDOM_SEED:-42}"
BANK_LIMIT="${BANK_LIMIT:-}"
MAX_TOKENS="${MAX_TOKENS:-256}"
OUTPUT_PATH="${OUTPUT_PATH:-result/GCG/gcg_suffix_bank_transfer_2c_heldout100.jsonl}"
APPEND_EOS_TAG="${APPEND_EOS_TAG:-1}"

cd "$PROJECT_DIR"

echo "[job] start $(date)"
echo "[job] host $(hostname)"
echo "[job] pwd $(pwd)"
echo "[job] id ${SLURM_JOB_ID:-manual}"
echo "[job] bank $BANK_PATH"
echo "[job] data $DATA_PATH"
echo "[job] template $TEMPLATE_PATH"
echo "[job] nclass $NCLASS"
echo "[job] max_samples $MAX_SAMPLES"
echo "[job] output $OUTPUT_PATH"

if [ ! -s "$BANK_PATH" ]; then
  echo "[error] suffix bank not found or empty: $BANK_PATH"
  echo "[hint] create it as one suffix per line, or pass BANK_PATH=/path/to/bank.jsonl"
  exit 2
fi

module load anaconda3
module load "$CUDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

ARGS=(
  --bank "$BANK_PATH"
  --data "$DATA_PATH"
  --template "$TEMPLATE_PATH"
  --nclass "$NCLASS"
  --max-samples "$MAX_SAMPLES"
  --random-seed "$RANDOM_SEED"
  --max-tokens "$MAX_TOKENS"
  --output "$OUTPUT_PATH"
)

if [ -n "$BANK_LIMIT" ]; then
  ARGS+=(--bank-limit "$BANK_LIMIT")
fi

if [ "$APPEND_EOS_TAG" = "1" ]; then
  ARGS+=(--append-eos-tag)
fi

python scripts/eval_gcg_suffix_bank.py "${ARGS[@]}"

echo "[job] done $(date)"
