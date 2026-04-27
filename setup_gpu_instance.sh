#!/usr/bin/env bash
set -e

echo "=== System update ==="
sudo apt update

echo "=== Installing base tools ==="
sudo apt install -y software-properties-common build-essential tmux git

echo "=== Adding deadsnakes PPA ==="
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

echo "=== Installing Python 3.11 ==="
sudo apt install -y python3.11 python3.11-venv python3.11-dev

echo "=== Creating projects directory ==="
mkdir -p ~/projects
cd ~/projects

echo "=== Cloning goal-composition repo (if needed) ==="
if [ ! -d "goal-composition" ]; then
  git clone https://github.com/diksha-shrivastava13/goal-composition.git
fi

cd goal-composition

echo "=== Creating virtual environment ==="
python3.11 -m venv env_pred
source env_pred/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing uv ==="
pip install uv

echo "=== Installing JAX GPU stack ==="
pip install jax==0.4.23 \
            jaxlib==0.4.23+cuda12.cudnn89 \
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "=== Installing NVIDIA CUDA libraries (pinned from working snapshot) ==="
pip install \
  nvidia-cublas-cu12==12.9.1.4 \
  nvidia-cuda-cupti-cu12==12.9.79 \
  nvidia-cuda-nvcc-cu12==12.9.86 \
  nvidia-cuda-nvrtc-cu12==12.9.86 \
  nvidia-cuda-runtime-cu12==12.9.79 \
  nvidia-cudnn-cu12==8.9.7.29 \
  nvidia-cufft-cu12==11.4.1.4 \
  nvidia-cusolver-cu12==11.7.5.82 \
  nvidia-cusparse-cu12==12.5.10.65 \
  nvidia-nccl-cu12==2.29.3 \
  nvidia-nvjitlink-cu12==12.9.86

echo "=== Installing project dependencies ==="
uv pip install -r requirements_no_jax.txt

echo "=== Re-pinning NVIDIA libs (in case requirements_no_jax.txt pulled different versions) ==="
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12==12.9.1.4 \
  nvidia-cuda-cupti-cu12==12.9.79 \
  nvidia-cuda-nvcc-cu12==12.9.86 \
  nvidia-cuda-nvrtc-cu12==12.9.86 \
  nvidia-cuda-runtime-cu12==12.9.79 \
  nvidia-cudnn-cu12==8.9.7.29 \
  nvidia-cufft-cu12==11.4.1.4 \
  nvidia-cusolver-cu12==11.7.5.82 \
  nvidia-cusparse-cu12==12.5.10.65 \
  nvidia-nccl-cu12==2.29.3 \
  nvidia-nvjitlink-cu12==12.9.86

echo "=== Persisting LD_LIBRARY_PATH in venv activation ==="
ACTIVATE_SCRIPT=~/projects/goal-composition/env_pred/bin/activate
if ! grep -q "CUDA libraries for JAX GPU" "$ACTIVATE_SCRIPT"; then
  # Use short variable to avoid line-wrap issues
  cat >> "$ACTIVATE_SCRIPT" << 'ENVFIX'

# CUDA libraries for JAX GPU
_SP="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia"
export LD_LIBRARY_PATH="$_SP/cudnn/lib:$_SP/cublas/lib:$_SP/cuda_runtime/lib:$_SP/cuda_cupti/lib:$_SP/cufft/lib:$_SP/cusolver/lib:$_SP/cusparse/lib:$_SP/nvjitlink/lib:$_SP/nccl/lib"
ENVFIX
fi

echo "=== Setting LD_LIBRARY_PATH for verification ==="
_SP="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia"
export LD_LIBRARY_PATH="$_SP/cudnn/lib:$_SP/cublas/lib:$_SP/cuda_runtime/lib:$_SP/cuda_cupti/lib:$_SP/cufft/lib:$_SP/cusolver/lib:$_SP/cusparse/lib:$_SP/nvjitlink/lib:$_SP/nccl/lib"

echo "=== Verifying GPU ==="
python - <<'PY'
import jax
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
assert jax.default_backend() == "gpu", "ERROR: JAX not using GPU!"
print("GPU verification passed.")
PY

echo "=== Environment ready ==="
echo "Activate later with:"
echo "  source ~/projects/goal-composition/env_pred/bin/activate"

echo "=== Manual steps ==="
echo "1) Activate env"
echo "2) Run: wandb login"
echo "3) Start experiment"