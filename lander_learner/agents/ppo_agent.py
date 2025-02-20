from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from lander_learner.utils.helpers import adjust_save_path, adjust_load_path
from lander_learner.utils.config import RL_Config
from lander_learner.agents.base_agent import BaseAgent
from lander_learner.agents import default_callback


class PPOAgent(BaseAgent):
    """An RL agent that uses Proximal Policy Optimization (PPO).

    This agent is implemented using the stable_baselines3 PPO algorithm.
    It checks the environment, sets up the model, and provides methods to train,
    predict actions, and save/load the model.
    """

    def __init__(self, env, deterministic=True, **kwargs):
        """Initializes a PPOAgent instance.

        Args:
            env: The Gym environment instance.
            deterministic (bool, optional): Whether the agent acts deterministically. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the PPO constructor.
        """
        # Only run check_env for non-vectorized environments.
        if not (isinstance(env, SubprocVecEnv) or isinstance(env, DummyVecEnv)):
            check_env(env, warn=True)
        super().__init__(env, deterministic)
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(RL_Config.DEFAULT_LOGGING_DIR / "lander_tensorboard"),
            **kwargs  # Extra arguments passed to PPO, if any.
        )

    def train(self, timesteps=RL_Config.CHECKPOINT_FREQ, callback=None):
        """Trains the PPO agent for a specified number of timesteps.

        Args:
            timesteps (int, optional): The total number of timesteps for training.
                Defaults to RL_Config.CHECKPOINT_FREQ.
            callback (optional): A callback function for saving checkpoints.
                Defaults to a default callback provided by the default_callback function.

        The model's learning process is logged and a tensorboard log is generated.
        """
        if callback is None:
            callback = default_callback(checkpoint_freq=timesteps, model_type="ppo")
        self.model.learn(
            total_timesteps=timesteps,
            tb_log_name="PPO_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
            callback=callback
        )

    def get_action(self, observation):
        """Returns an action given an observation using the PPO model.

        Args:
            observation: The observation input to the model.

        Returns:
            The predicted action.
        """
        action, _states = self.model.predict(observation, deterministic=self.deterministic)
        return action

    def save_model(self, path):
        """Saves the PPO model to the specified path.

        Args:
            path (str): The file path or directory where the model will be saved.
        """
        path = adjust_save_path(path, model_type="ppo")
        self.model.save(path)

    def load_model(self, path):
        """Loads the PPO model from the specified path.

        Args:
            path (str): The file path or directory from which to load the model.
        """
        path = adjust_load_path(path, model_type="ppo")
        self.model = PPO.load(path, env=self.env)
