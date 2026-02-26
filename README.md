## Goal Composition

⚠️ **Research repository in active development.** Components may be refactored as experiments scale.

🧪 Environment setup instructions are included for reproducibility.

This repository contains ongoing research on goal composition, agentic traits, and environmental induction of behavior in reinforcement learning agents. The codebase supports large-scale experimentation with JAX-based training in procedurally generated environments.

Structure, experiments, and dependencies may evolve as the project progresses.

---

### 🧠 Research Statement

* Can we model what shards a particular environment induces?
* How do training dynamics causally produce inner goal structures?

---

### ⭐ Reproducibility

Experiments in this repository were conducted using a GPU-based environment with a manually installed JAX CUDA stack.
Due to the sensitivity of JAX GPU builds to driver and CUDA versions, installation is performed in two phases.

---

#### 📦 Reference Environment

The exact environment used for experiments is documented in `requirements_gpu_snapshot.txt`.

Header (abridged):

* Instance: g6.2xlarge
* GPU: NVIDIA L4
* Driver: 580.126.09
* CUDA (driver): 13.0
* cuDNN: 8.9.7
* JAX: 0.4.23 + cuda12.cudnn89
* Python: 3.11.14
* OS: Ubuntu 24.04.4 LTS

This snapshot enables reconstruction of the full working environment if needed.

---

#### 🧩 Dependency Structure

Dependencies are split into three files for stability and reproducibility.

##### 1. requirements.txt (CPU / local prototyping)

Used for local development, CPU-only experiments, and rapid prototyping.

```
uv pip install -r requirements.txt
```

---

##### 2. requirements_no_jax.txt (GPU project dependencies)

Contains all project dependencies **excluding JAX and CUDA packages**.
Used after manual installation of the JAX GPU stack via the setup script.

```
uv pip install -r requirements_no_jax.txt
```

---

##### 3. requirements_gpu_snapshot.txt (full environment snapshot)

A frozen record of the exact environment used in experiments, including JAX, CUDA runtime packages, and all transitive dependencies.
This file is for provenance and reproducibility, not routine installation.

---

### 🛠️ GPU Setup (Recommended)

Use the provided setup script to configure a compatible environment on a fresh instance.

**Setup Script:** `setup_gpu_instance.sh`

The script performs:

* System update
* Installation of Python 3.11
* Virtual environment creation
* Manual JAX GPU installation
* cuDNN installation
* CUDA library path configuration
* Installation of project dependencies
* GPU backend verification

Run:

```
bash setup_instance.sh
```

After completion:

```
source ~/projects/goal-composition/env_pred/bin/activate
wandb login
```

You can then begin training or experimentation.

---

#### ✅ Verification

Confirm GPU backend:

```
python - <<'PY'
import jax
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
PY
```

Expected output:

```
Backend: gpu
Devices: [cuda(id=0)]
```

---

### 🧭 Research Notes (Ongoing)

The following notes describe current research directions and may change as work progresses.

#### Research Plan (04.12.2025)

1. Training agents robust to environmental interventions in Minigrid variants using JaxUED
2. Theoretical work with Causal Influence Diagrams for inferring intention and instrumental goals
3. Chess puzzles as a testbed for composition of agentic traits

These notes document hypotheses, directions, and open questions rather than finalized conclusions.

---

### 🙏 Acknowledgements

This project builds on prior work in procedural environment design and relies extensively on JaxUED.
Thanks to the original developers for the library and their research on goal misgeneralisation in procedurally generated environments.
