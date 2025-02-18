from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from lander_learner.utils.helpers import adjust_save_path, adjust_load_path
from lander_learner.utils.rl_config import RL_Config
from lander_learner.agents.base_agent import BaseAgent
from lander_learner.agents import default_callback


class SACAgent(BaseAgent):
    """
    An RL agent that uses Soft Actor-Critic (SAC) with GPU acceleration where available.
    """

    def __init__(self, env, deterministic=True, device="auto", **kwargs):
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
            device=self.device,  # Uses GPU if available: "cuda" or "auto"
            **kwargs  # Extra arguments passed to SAC, if any
        )

    def train(self, timesteps=100000, callback=None, checkpoint_freq=RL_Config.CHECKPOINT_FREQ):
        if callback is None:
            callback = default_callback(checkpoint_freq=checkpoint_freq, model_type="sac")
        self.model.learn(
            total_timesteps=timesteps,
            tb_log_name="SAC_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
            callback=callback
        )

    def get_action(self, observation):
        action, _states = self.model.predict(observation, deterministic=self.deterministic)
        return action

    def save_model(self, path):
        path = adjust_save_path(path, model_type="sac")
        self.model.save(path)

    def load_model(self, path):
        path = adjust_load_path(path, model_type="sac")
        self.model = SAC.load(path, env=self.env)
