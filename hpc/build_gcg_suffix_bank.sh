#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH -t 1-00:00:00
#SBATCH -J build_bank
#SBATCH -o /dev/null
#SBATCH -e /dev/null

PROJECT_DIR="${PROJECT_DIR:-$HOME/GradingAttack}"
LOG_DIR="$PROJECT_DIR/hpc/logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/build_gcg_bank_${SLURM_JOB_ID:-manual}.out" 2> "$LOG_DIR/build_gcg_bank_${SLURM_JOB_ID:-manual}.err"

set -eo pipefail

ENV_NAME="${ENV_NAME:-gradingattack}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"
DATA_PATH="${DATA_PATH:-dataset/scientsbank.jsonl}"
TEMPLATE_PATH="${TEMPLATE_PATH:-configs/grading_template_ci_2c.txt}"
NCLASS="${NCLASS:-2}"
SUCCESS_MODE="${SUCCESS_MODE:-correct}"
MAX_CANDIDATES="${MAX_CANDIDATES:-50}"
TARGET_BANK_SIZE="${TARGET_BANK_SIZE:-10}"
RANDOM_SEED="${RANDOM_SEED:-42}"
NUM_STEPS="${NUM_STEPS:-100}"
SEARCH_WIDTH="${SEARCH_WIDTH:-64}"
TOPK="${TOPK:-64}"
MAX_TOKENS="${MAX_TOKENS:-256}"
OUTPUT_BANK="${OUTPUT_BANK:-gcg_suffix_bank.jsonl}"
ATTEMPT_LOG="${ATTEMPT_LOG:-result/GCG/gcg_suffix_bank_build_attempts.jsonl}"
RUN_TRANSFER_AFTER="${RUN_TRANSFER_AFTER:-1}"
TRANSFER_MAX_SAMPLES="${TRANSFER_MAX_SAMPLES:-100}"
TRANSFER_OUTPUT="${TRANSFER_OUTPUT:-result/GCG/gcg_suffix_bank_transfer_after_build.jsonl}"
APPEND_EOS_TAG="${APPEND_EOS_TAG:-1}"

cd "$PROJECT_DIR"

echo "[job] start $(date)"
echo "[job] host $(hostname)"
echo "[job] pwd $(pwd)"
echo "[job] id ${SLURM_JOB_ID:-manual}"
echo "[job] data $DATA_PATH"
echo "[job] template $TEMPLATE_PATH"
echo "[job] nclass $NCLASS"
echo "[job] success_mode $SUCCESS_MODE"
echo "[job] max_candidates $MAX_CANDIDATES"
echo "[job] target_bank_size $TARGET_BANK_SIZE"
echo "[job] output_bank $OUTPUT_BANK"
echo "[job] attempt_log $ATTEMPT_LOG"
echo "[job] run_transfer_after $RUN_TRANSFER_AFTER"
echo "[job] transfer_output $TRANSFER_OUTPUT"

module load anaconda3
module load "$CUDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python -c "import nanogcg, transformers, yaml; print('imports ok')"

ARGS=(
  --data "$DATA_PATH"
  --template "$TEMPLATE_PATH"
  --nclass "$NCLASS"
  --success-mode "$SUCCESS_MODE"
  --max-candidates "$MAX_CANDIDATES"
  --target-bank-size "$TARGET_BANK_SIZE"
  --random-seed "$RANDOM_SEED"
  --num-steps "$NUM_STEPS"
  --search-width "$SEARCH_WIDTH"
  --topk "$TOPK"
  --max-tokens "$MAX_TOKENS"
  --output-bank "$OUTPUT_BANK"
  --attempt-log "$ATTEMPT_LOG"
)

if [ "$APPEND_EOS_TAG" = "1" ]; then
  ARGS+=(--append-eos-tag)
fi

python scripts/build_gcg_suffix_bank.py "${ARGS[@]}"

if [ "$RUN_TRANSFER_AFTER" = "1" ] && [ -s "$OUTPUT_BANK" ]; then
  echo "[job] transfer start $(date)"
  TRANSFER_ARGS=(
    --bank "$OUTPUT_BANK"
    --data "$DATA_PATH"
    --template "$TEMPLATE_PATH"
    --nclass "$NCLASS"
    --max-samples "$TRANSFER_MAX_SAMPLES"
    --random-seed "$RANDOM_SEED"
    --max-tokens "$MAX_TOKENS"
    --output "$TRANSFER_OUTPUT"
  )
  if [ "$APPEND_EOS_TAG" = "1" ]; then
    TRANSFER_ARGS+=(--append-eos-tag)
  fi
  python scripts/eval_gcg_suffix_bank.py "${TRANSFER_ARGS[@]}"
else
  echo "[job] transfer skipped; RUN_TRANSFER_AFTER=$RUN_TRANSFER_AFTER or empty bank: $OUTPUT_BANK"
fi

echo "[job] done $(date)"
