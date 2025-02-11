import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from utils.helpers import adjust_save_path, adjust_load_path

class RLAgent:
    """
    An RL-based agent utilizing Stable-Baselines3 (e.g. PPO).
    """

    def __init__(self, env):
        """
        env: A Gymnasium environment.
        """
        if not hasattr(env, "envs"):
            check_env(env, warn=True)  # Checks if the env follows Gym API
        self.env = env
        self.model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log="./agents/logs/ppo_lunar_lander_tensorboard/"
            )

    def train(self, timesteps=10000):
        """
        Train the RL model for a specified number of timesteps.
        """
        self.model.learn(total_timesteps=timesteps)

    def get_action(self, observation):
        """
        Predict an action given the current observation.
        """
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def save_model(self, path):
        path = adjust_save_path(path)
        self.model.save(path)

    def load_model(self, path):
        path = adjust_load_path(path)
        self.model = PPO.load(path, env=self.env)
