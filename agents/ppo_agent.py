import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from utils.helpers import adjust_save_path, adjust_load_path
from utils.config import Config
from agents.base_agent import BaseAgent

class PPOAgent(BaseAgent):
    """
    An RL agent that uses Proximal Policy Optimization (PPO).
    """
    def __init__(self, env, deterministic=True, **kwargs):
        # Only run check_env for non-vectorized environments.
        if not (isinstance(env, SubprocVecEnv) or isinstance(env, DummyVecEnv)):
            check_env(env, warn=True)
        super().__init__(env, deterministic)
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(Config.DEFAULT_LOGGING_DIR / "ppo_lander_tensorboard"),
            **kwargs  # Extra arguments passed to PPO, if any
        )

    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)

    def get_action(self, observation):
        action, _states = self.model.predict(observation, deterministic=self.deterministic)
        return action

    def save_model(self, path):
        path = adjust_save_path(path, model_type="ppo")
        self.model.save(path)

    def load_model(self, path):
        path = adjust_load_path(path, model_type="ppo")
        self.model = PPO.load(path, env=self.env)