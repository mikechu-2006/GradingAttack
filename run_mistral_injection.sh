#!/bin/bash
# Run Mistral-7B-Instruct-v0.3 injection attacks (all 4 variants)
# Usage: bash run_mistral_injection.sh
# Use with --pipeline flag via main.py (local) or sbatch (HPC)

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

CONFIGS=(
  "configs/RolePlay-Mistral-7B-Instruct-v0.3.yaml"
  "configs/injection/ao-Mistral-7B-Instruct-v0.3-2c.yaml"
  "configs/injection/dc-Mistral-7B-Instruct-v0.3-2c.yaml"
  "configs/injection/im-Mistral-7B-Instruct-v0.3-2c.yaml"
)

echo "=========================================="
echo " Mistral-7B-Instruct-v0.3 Injection Attacks"
echo "=========================================="
echo ""

for cfg in "${CONFIGS[@]}"; do
  name=$(grep -m1 '^name:' "$cfg" | sed 's/name: //')
  echo "[run] $name"
  echo "  config: $cfg"
  echo "  start: $(date)"

  python main.py "$cfg" --pipeline

  echo "  done:  $(date)"
  echo ""
done

echo "=========================================="
echo " All 4 runs completed at $(date)"
echo "=========================================="
