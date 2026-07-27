#!/bin/bash
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/GradingAttack}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
RANDOM_SEED="${RANDOM_SEED:-42}"
BANK_PATH="${BANK_PATH:-result/GCG/gcg_suffix_bank_3c_promotion.jsonl}"
RUN_SCRIPT="${RUN_SCRIPT:-hpc/run_merged_pipeline.sh}"

# Defaults add four models beyond the existing Llama-3.1-8B run, giving five
# total models for the attention-detector study.
MODEL_SPECS="${MODEL_SPECS:-\
qwen25-7b|Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct|
qwen3-4b-2507|Qwen3-4B-Instruct-2507|Qwen/Qwen3-4B-Instruct-2507|
qwen35-4b|Qwen3.5-4B|Qwen/Qwen3.5-4B|
qwen25-3b|Qwen2.5-3B-Instruct|Qwen/Qwen2.5-3B-Instruct|}"

cd "$PROJECT_DIR"
mkdir -p configs/generated

echo "[submit] project: $PROJECT_DIR"
echo "[submit] bank:    $BANK_PATH"
echo "[submit] samples: $MAX_SAMPLES"

while IFS='|' read -r tag model_name model_id model_path; do
  [ -n "$tag" ] || continue
  cfg="configs/generated/Pipeline-GCG-SuffixBank-${model_name}-3c-heldout${MAX_SAMPLES}-attention.yaml"
  echo "[submit] write $cfg"

  if [ -n "$model_path" ]; then
    model_block="  path: \"${model_path}\""
  else
    model_block="  model_id: \"${model_id}\""
  fi

  cat > "$cfg" <<EOF
name: Pipeline-GCG-SuffixBank-${model_name}-3c-heldout${MAX_SAMPLES}-attention

method: gcg_suffix_bank
pipeline_mode: true
debug: false
log_attention: true
nclass: 3
template: ci

params:
  bank_path: ${BANK_PATH}
  bank_limit: null
  exclude_bank_source_indices: true

model:
  name: ${model_name}
${model_block}

data:
  - name: scientsbank
    path: ./dataset/scientsbank.jsonl
    max_samples: ${MAX_SAMPLES}
    random_seed: ${RANDOM_SEED}

generation:
  temperature: 0.01
  max_tokens: 1024

log:
  log_dir: ./logs
  result_dir: ./result
EOF

  echo "[submit] sbatch $tag"
  sbatch --export=ALL,CONFIG_PATH="$cfg" "$RUN_SCRIPT"
done <<< "$MODEL_SPECS"
