#!/bin/bash
set -eo pipefail

MODEL_TAG="${MODEL_TAG:-qwen3-4b}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-}"
PHASE="${PHASE:-all}"
ALLOW_PARTIAL_TUNE="${ALLOW_PARTIAL_TUNE:-1}"

scripts=(
  run_pipeline_injection_ao_as.sh
  run_pipeline_injection_ao_hs.sh
  run_pipeline_injection_dc_as.sh
  run_pipeline_injection_dc_hs.sh
  run_pipeline_injection_im_as.sh
  run_pipeline_injection_im_hs.sh
)

echo "[submit] model tag: ${MODEL_TAG}"
echo "[submit] model name: ${MODEL_NAME}"
echo "[submit] model id:   ${MODEL_ID}"
echo "[submit] phase:      ${PHASE}"

for script in "${scripts[@]}"; do
  if [ ! -f "$script" ]; then
    echo "[submit] missing script: $script" >&2
    exit 1
  fi
  echo "[submit] sbatch $script"
  sbatch --export=ALL,MODEL_TAG="$MODEL_TAG",MODEL_NAME="$MODEL_NAME",MODEL_ID="$MODEL_ID",MODEL_PATH="$MODEL_PATH",PHASE="$PHASE",ALLOW_PARTIAL_TUNE="$ALLOW_PARTIAL_TUNE" "$script"
done
