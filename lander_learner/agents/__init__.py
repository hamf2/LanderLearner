from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
from lander_learner.utils.config import RL_Config


def default_callback(checkpoint_freq=100000, checkpoint_dir=None, model_type="model"):
    """
    Create a default CheckpointCallback for training an RL agent.

    Args:
        timesteps (int): The number of timesteps between each checkpoint.
        checkpoint_dir (str): The directory to save checkpoints.

    Returns:
        CheckpointCallback: A callback that saves checkpoints every `timesteps` steps.
    """
    if checkpoint_dir is None:
        checkpoint_dir = RL_Config.DEFAULT_CHECKPOINT_DIR / f"{model_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_dir, name_prefix=model_type)
