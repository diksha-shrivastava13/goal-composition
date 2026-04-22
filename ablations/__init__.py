"""
Curriculum Awareness Ablation Study

This package contains the implementation for systematically studying how RL agents
develop awareness of their training curriculum under different memory mechanisms.

5 Agent Architectures:
1. next_env_prediction - Explicit curriculum info baseline; a prediction head for env variables at every checkpoint.
2. accel_probe - No memory baseline (reset on episode). Base Agent, no advantage or leak at all.
3. persistent_lstm - Non-resetting LSTM (emergence test)
4. context_vector - Compressed EMA context
5. episodic_memory - Discrete episode buffer

38 Interpretability Experiments (16 universal + 22 PAIRED-specific):
See ablations/experiments/run_experiment.py for the complete registry.
"""

__version__ = "0.1.0"