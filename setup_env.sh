#!/bin/bash
set -e

ENV_NAME="isles22_ensemble"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n $ENV_NAME python=3.8.0 pip -y

echo "=== Activating environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "=== Installing uv ==="
pip install uv

echo "=== Installing PyTorch with CUDA 11.3 via uv (PyTorch index) ==="
cd "$SCRIPT_DIR"
uv pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0

echo "=== Installing all other dependencies via uv ==="
uv pip install -e .

echo "=== Setting up PYTHONPATH ==="
# Add to conda activate script for persistence
ACTIVATE_SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
mkdir -p "$(dirname "$ACTIVATE_SCRIPT")"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
export PYTHONPATH="\${PYTHONPATH}:${SCRIPT_DIR}/src/SEALS:${SCRIPT_DIR}/src/FACTORIZER/model/factorizer:${SCRIPT_DIR}/src/HD-BET"
EOF
chmod +x "$ACTIVATE_SCRIPT"

echo ""
echo "=== Done! ==="
echo "Activate the environment with: conda activate $ENV_NAME"
echo "The PYTHONPATH is automatically set when you activate the environment."

