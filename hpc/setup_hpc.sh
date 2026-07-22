#!/bin/bash
set -eo pipefail

ENV_NAME="${ENV_NAME:-gradingattack}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

cd "$(dirname "$0")/.."

echo "[setup] Workdir: $(pwd)"
echo "[setup] Loading modules..."
module load anaconda3
if module avail "$CUDA_MODULE" >/dev/null 2>&1; then
  module load "$CUDA_MODULE"
else
  echo "[setup] Warning: module $CUDA_MODULE not found by module avail; trying module load anyway."
  module load "$CUDA_MODULE"
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] Creating conda env: $ENV_NAME"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
else
  echo "[setup] Conda env already exists: $ENV_NAME"
fi

conda activate "$ENV_NAME"

echo "[setup] Python: $(which python)"
python --version
pip --version

echo "[setup] Installing baseline dependencies..."
pip install -U pip
pip install pyyaml scikit-learn transformers modelscope

echo "[setup] Installing PyTorch from $TORCH_INDEX_URL ..."
pip install torch==2.4.0 --index-url "$TORCH_INDEX_URL" || \
  pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

echo "[setup] Verifying imports..."
python - <<'PY'
import importlib.util as u
pkgs = ['yaml', 'sklearn', 'transformers', 'modelscope', 'torch']
print({p: bool(u.find_spec(p)) for p in pkgs})
import torch
print('torch', torch.__version__)
print('torch cuda', torch.version.cuda)
print('cuda available on this node', torch.cuda.is_available())
print('device count on this node', torch.cuda.device_count())
PY

echo "[setup] Done. Next: sbatch hpc/test_gpu.sh"

