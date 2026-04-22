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

### 🤸‍♀️ Repository Structure

This is the repository structure, most of the current active work (March 2026) is under ablations/.

```
├── ablations
│   ├── __init__.py
│   ├── agents
│   │   ├── __init__.py
│   │   ├── accel_probe.py
│   │   ├── base.py
│   │   ├── context_vector.py
│   │   ├── episodic_memory.py
│   │   ├── next_env_prediction.py
│   │   ├── paired_accel_probe.py
│   │   ├── paired_base.py
│   │   ├── paired_context_vector.py
│   │   ├── paired_episodic_memory.py
│   │   ├── paired_next_env_prediction.py
│   │   ├── paired_persistent_lstm.py
│   │   └── persistent_lstm.py
│   ├── common
│   │   ├── __init__.py
│   │   ├── curriculum_state.py
│   │   ├── environment.py
│   │   ├── metrics.py
│   │   ├── networks.py
│   │   ├── probe_runner.py
│   │   ├── training.py
│   │   ├── types.py
│   │   ├── utils.py
│   │   └── visualization.py
│   ├── configs
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── defaults.py
│   │   ├── experiment_defaults.py
│   │   ├── presets.py
│   │   └── README.md
│   ├── experiments
│   │   ├── __init__.py
│   │   ├── activation_analysis.py
│   │   ├── base.py
│   │   ├── behavioral_coupling.py
│   │   ├── causal_intervention.py
│   │   ├── counterfactual.py
│   │   ├── cross_agent_comparison.py
│   │   ├── cross_episode_flow.py
│   │   ├── dr_coverage.py
│   │   ├── goal_extraction.py
│   │   ├── level_probing.py
│   │   ├── mutation_adaptation.py
│   │   ├── n_env_prediction.py
│   │   ├── n_step_prediction.py
│   │   ├── output_probing.py
│   │   ├── paired
│   │   │   ├── __init__.py
│   │   │   ├── activation_patching.py
│   │   │   ├── adversary_ablation.py
│   │   │   ├── adversary_dynamics.py
│   │   │   ├── adversary_policy_extraction.py
│   │   │   ├── adversary_strategy_clustering.py
│   │   │   ├── antagonist_audit.py
│   │   │   ├── belief_behaviour_divergence.py
│   │   │   ├── belief_revision_detection.py
│   │   │   ├── bilateral_utility.py
│   │   │   ├── causal_model_extraction.py
│   │   │   ├── coalition_dynamics.py
│   │   │   ├── counterfactual_curriculum.py
│   │   │   ├── goal_evolution.py
│   │   │   ├── multiscale_goals.py
│   │   │   ├── regret_decomposition.py
│   │   │   ├── regret_transfer.py
│   │   │   ├── representation_divergence.py
│   │   │   ├── representation_trajectory.py
│   │   │   ├── shard_dynamics.py
│   │   │   ├── teaching_opacity.py
│   │   │   ├── teaching_signal_intervention.py
│   │   │   └── utility_extraction.py
│   │   ├── phase_transition.py
│   │   ├── probes
│   │   │   ├── __init__.py
│   │   │   ├── output_probe.py
│   │   │   └── property_probe.py
│   │   ├── run_all.py
│   │   ├── run_experiment.py
│   │   ├── run_paired_suite.py
│   │   ├── symbolic_regression.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── activation_patching.py
│   │   │   ├── agent_aware_loss.py
│   │   │   ├── batched_rollout.py
│   │   │   ├── calibration_utils.py
│   │   │   ├── distribution_shift.py
│   │   │   ├── history_injection.py
│   │   │   ├── memory_probing.py
│   │   │   ├── paired_helpers.py
│   │   │   ├── rsa_cka.py
│   │   │   ├── time_series_analysis.py
│   │   │   ├── transfer_metrics.py
│   │   │   └── transition_metrics.py
│   │   └── value_calibration.py
│   ├── memory
│   │   ├── __init__.py
│   │   ├── context_vector.py
│   │   ├── episodic_buffer.py
│   │   ├── persistent_rnn.py
│   │   └── reset_rnn.py
│   └── scripts
│       ├── __init__.py
│       ├── comprehensive_experiment_test.py
│       ├── evaluate.py
│       ├── README.md
│       ├── shell
│       │   ├── run_matrix.sh
│       │   └── train_all.sh
│       ├── smoke_test.py
│       ├── test_paired_experiments.py
│       ├── train_all.py
│       ├── train_with_experiments.py
│       └── train.py
├── docs
├── elicting_world_models
│   ├── __init__.py
│   └── accel_minigrid_cwm.py
├── LICENSE
├── misgeneralisation
│   ├── __init__.py
│   └── keys_chests_accel.py
├── originals
│   ├── original_accel.py
│   ├── original_dr.py
│   └── original_paired.py
├── README.md
├── requirements_dev_local.txt
├── requirements_gpu_snapshot.txt
├── requirements_no_jax.txt
├── scalable_oversight
│   ├── __init__.py
│   ├── accel_probe.py
│   ├── nep_dr.py
│   ├── next_env_prediction.py
│   └── prediction_head_ablations.py
└── setup_gpu_instance.sh

```

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
