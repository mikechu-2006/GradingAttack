#!/bin/bash
set -eo pipefail

ENV_NAME="${ENV_NAME:-gradingattack}"

# Llama has already been run locally/on HPC in our main result. These defaults
# download the four additional models so the later GPU jobs do not waste GPU
# allocation time on network transfer.
MODEL_SPECS="${MODEL_SPECS:-\
qwen25-7b|Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct|
qwen3-4b-2507|Qwen3-4B-Instruct-2507|Qwen/Qwen3-4B-Instruct-2507|
qwen35-4b|Qwen3.5-4B|Qwen/Qwen3.5-4B|
qwen25-3b|Qwen2.5-3B-Instruct|Qwen/Qwen2.5-3B-Instruct|}"
export MODEL_SPECS

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python - <<'PY'
import os
import sys
from modelscope import snapshot_download

specs = [line.strip() for line in os.environ["MODEL_SPECS"].splitlines() if line.strip()]
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
failed = []

for spec in specs:
    parts = (spec.split("|") + ["", "", "", ""])[:4]
    tag, name, model_id, model_path = parts
    if model_path:
        print(f"[download] skip {tag}: explicit local path {model_path}", flush=True)
        continue
    print(f"[download] {tag}: {model_id}", flush=True)
    try:
        path = snapshot_download(model_id, cache_dir=cache_dir)
        print(f"[download] ok {tag}: {path}", flush=True)
    except Exception as exc:
        print(f"[download] FAIL {tag}: {exc!r}", flush=True)
        failed.append((tag, model_id))

if failed:
    print("[download] failed models:", failed, flush=True)
    sys.exit(1)
PY

