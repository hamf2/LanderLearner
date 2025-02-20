"""
Agents Package

This package contains implementations of RL agents for the Lunar Lander environment.
It provides a base class (BaseAgent) and concrete implementations using different algorithms,
such as:
  - HumanAgent: For manual control via keyboard input.
  - PPOAgent: An agent using Proximal Policy Optimization.
  - SACAgent: An agent using Soft Actor-Critic.

It also includes helper functions for default checkpoint callbacks.
"""

from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
from lander_learner.utils.config import RL_Config


def default_callback(checkpoint_freq=100000, checkpoint_dir=None, model_type="model"):
    """Creates a default checkpoint callback for training an RL agent.

    This function returns a CheckpointCallback configured to save the model every
    `checkpoint_freq` timesteps. If no checkpoint directory is provided, a default
    directory is generated based on the model type and the current datetime.

    Args:
        checkpoint_freq (int, optional): Number of timesteps between each checkpoint.
            Defaults to 100000.
        checkpoint_dir (str, optional): Directory to save checkpoints. If None, a default path is used.
        model_type (str, optional): A string representing the type of model (e.g., "ppo", "sac").
            Defaults to "model".

    Returns:
        CheckpointCallback: A callback for saving model checkpoints.
    """
    if checkpoint_dir is None:
        checkpoint_dir = RL_Config.DEFAULT_CHECKPOINT_DIR / f"{model_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_dir, name_prefix=model_type)
