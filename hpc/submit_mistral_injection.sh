#!/bin/bash
# Submit Mistral-7B-Instruct-v0.3 injection attack jobs to HPC
# Usage: bash hpc/submit_mistral_injection.sh
#
# Overrides:
#   MAX_SAMPLES=20 bash hpc/submit_mistral_injection.sh   # smoke test

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/GradingAttack}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
RANDOM_SEED="${RANDOM_SEED:-42}"
RUN_SCRIPT="${RUN_SCRIPT:-hpc/run_merged_pipeline.sh}"

cd "$PROJECT_DIR"

CONFIGS=(
  "configs/RolePlay-Mistral-7B-Instruct-v0.3.yaml"
  "configs/injection/ao-Mistral-7B-Instruct-v0.3-2c.yaml"
  "configs/injection/dc-Mistral-7B-Instruct-v0.3-2c.yaml"
  "configs/injection/im-Mistral-7B-Instruct-v0.3-2c.yaml"
)

echo "[submit] project: $PROJECT_DIR"
echo "[submit] samples: $MAX_SAMPLES"
echo "[submit] jobs:    ${#CONFIGS[@]}"
echo ""

# Override max_samples in temp configs so the originals stay clean
mkdir -p configs/generated

for cfg in "${CONFIGS[@]}"; do
  name=$(grep -m1 '^name:' "$cfg" | sed 's/name: //')
  tag=$(basename "$cfg" .yaml)
  gen_cfg="configs/generated/${tag}-mistral-${MAX_SAMPLES}.yaml"
  label="mistral_${tag}"

  # Copy and patch max_samples if needed
  cp "$cfg" "$gen_cfg"
  if grep -q "max_samples:" "$gen_cfg"; then
    sed -i "s/max_samples:.*/max_samples: ${MAX_SAMPLES}/" "$gen_cfg"
  else
    sed -i "/^data:/a\    max_samples: ${MAX_SAMPLES}" "$gen_cfg"
  fi

  echo "[submit] $tag  ->  sbatch $label"
  sbatch --export=ALL,CONFIG_PATH="$gen_cfg" \
         --job-name="$label" \
         "$RUN_SCRIPT"
done

echo ""
echo "[submit] All jobs submitted. Check with: squeue -u \$USER"
