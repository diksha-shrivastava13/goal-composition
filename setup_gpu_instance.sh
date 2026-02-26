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

echo "=== Fixing NumPy compatibility ==="
pip install numpy==1.26.4

echo "=== Installing matching cuDNN ==="
pip uninstall -y nvidia-cudnn-cu12 || true
pip install nvidia-cudnn-cu12==8.9.7.29

echo "=== Setting CUDA library paths ==="
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cublas/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cuda_runtime/lib

echo "=== Installing project dependencies (no JAX) ==="
uv pip install -r requirements_no_jax.txt

echo "=== Verifying GPU ==="
python - <<'PY'
import jax
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
PY

echo "=== Environment ready ==="
echo "Activate later with:"
echo "source ~/projects/goal-composition/env_pred/bin/activate"

echo "=== Manual steps ==="
echo "1) Activate env"
echo "2) Run: wandb login"
echo "3) Start experiment"