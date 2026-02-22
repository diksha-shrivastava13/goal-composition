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

9 Interpretability Experiments:
1. Level property probing
2. Value calibration
3. Mutation adaptation
4. Causal intervention
5. Weights/activations analysis
6. Policy/value output probing
7. Symbolic regression (U_hat extraction)
8. Behavioral coupling
9. Counterfactual intervention
"""

# TODO: Update the list of experiments here, have far exceeded.

__version__ = "0.1.0"