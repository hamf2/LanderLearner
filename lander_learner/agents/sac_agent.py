from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from lander_learner.utils.helpers import adjust_save_path, adjust_load_path
from lander_learner.utils.config import RL_Config
from lander_learner.agents.base_agent import BaseAgent
from lander_learner.agents import default_callback


class SACAgent(BaseAgent):
    """An RL agent that uses Soft Actor-Critic (SAC) with GPU acceleration if available.

    This agent is implemented using the stable_baselines3 SAC algorithm.
    It performs environment checking, sets up the SAC model, and provides methods to train,
    predict actions, and save/load the model.
    """

    def __init__(self, env, deterministic=True, device="auto", **kwargs):
        """Initializes a SACAgent instance.

        Args:
            env: The Gym environment instance.
            deterministic (bool, optional): Whether the agent acts deterministically. Defaults to True.
            device (str, optional): The device to use ("auto", "cpu", or "cuda"). Defaults to "auto".
            **kwargs: Additional keyword arguments to pass to the SAC constructor.
        """
        # Only run check_env for non-vectorized environments.
        if not (isinstance(env, SubprocVecEnv) or isinstance(env, DummyVecEnv)):
            check_env(env, warn=True)
        super().__init__(env, deterministic)
        self.device = device
        self.model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(RL_Config.DEFAULT_LOGGING_DIR / "lander_tensorboard"),
            device=self.device,  # Uses GPU if available.
            **kwargs  # Extra arguments passed to SAC.
        )

    def train(self, timesteps=100000, callback=None, checkpoint_freq=RL_Config.CHECKPOINT_FREQ):
        """Trains the SAC agent for a specified number of timesteps.

        Args:
            timesteps (int, optional): The total number of timesteps for training.
                Defaults to 100000.
            callback (optional): A callback function for checkpointing during training.
                Defaults to a callback provided by default_callback.
            checkpoint_freq (int, optional): Frequency of checkpoints. Defaults to RL_Config.CHECKPOINT_FREQ.

        The model's learning is logged and monitored via TensorBoard.
        """
        if callback is None:
            callback = default_callback(checkpoint_freq=checkpoint_freq, model_type="sac")
        self.model.learn(
            total_timesteps=timesteps,
            tb_log_name="SAC_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
            callback=callback
        )

    def get_action(self, observation):
        """Returns an action given an observation using the SAC model.

        Args:
            observation: The observation input to the model.

        Returns:
            The predicted action.
        """
        action, _states = self.model.predict(observation, deterministic=self.deterministic)
        return action

    def save_model(self, path):
        """Saves the SAC model to the specified path.

        Args:
            path (str): The file path or directory where the model will be saved.
        """
        path = adjust_save_path(path, model_type="sac")
        self.model.save(path)

    def load_model(self, path):
        """Loads the SAC model from the specified path.

        Args:
            path (str): The file path or directory from which to load the model.
        """
        path = adjust_load_path(path, model_type="sac")
        self.model = SAC.load(path, env=self.env)
