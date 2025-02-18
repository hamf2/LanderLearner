from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from lander_learner.utils.helpers import adjust_save_path, adjust_load_path
from lander_learner.utils.rl_config import RL_Config
from lander_learner.agents.base_agent import BaseAgent
from lander_learner.agents import default_callback


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
            tensorboard_log=str(RL_Config.DEFAULT_LOGGING_DIR / "lander_tensorboard"),
            **kwargs  # Extra arguments passed to PPO, if any
        )

    def train(self, timesteps=RL_Config.CHECKPOINT_FREQ, callback=None):
        if callback is None:
            callback = default_callback(checkpoint_freq=timesteps, model_type="ppo")
        self.model.learn(
            total_timesteps=timesteps,
            tb_log_name="PPO_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
            callback=callback
        )

    def get_action(self, observation):
        action, _states = self.model.predict(observation, deterministic=self.deterministic)
        return action

    def save_model(self, path):
        path = adjust_save_path(path, model_type="ppo")
        self.model.save(path)

    def load_model(self, path):
        path = adjust_load_path(path, model_type="ppo")
        self.model = PPO.load(path, env=self.env)
