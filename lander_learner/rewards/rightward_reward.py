import numpy as np
from lander_learner.rewards.base_reward import BaseReward
from lander_learner.utils.config import Config
from lander_learner.utils.rl_config import RL_Config
import logging

logger = logging.getLogger(__name__)


class RightwardReward(BaseReward):
    def __init__(self, **kwargs):
        """
        Initialize DefaultReward with configurable parameters.

        Possible keyword arguments:
            x_velocity_factor (float): Factor for rewarding rightward velocity.
                                        Default: RL_Config.DEFAULT_DEFAULT_REWARD_PARAMS["x_velocity_factor"]
            angle_penalty_factor (float): Factor for penalizing deviation from π/2.
                                          Default: RL_Config.DEFAULT_DEFAULT_REWARD_PARAMS["angle_penalty_factor"]
            collision_penalty (float): Penalty per time step when a collision is detected.
                                       Default: RL_Config.DEFAULT_DEFAULT_REWARD_PARAMS["collision_penalty"]
            crash_penalty_multiplier (float): Multiplier for penalty based on collision impulse on termination.
                                              Default: RL_Config.DEFAULT_DEFAULT_REWARD_PARAMS["
                                              crash_penalty_multiplier"]
        """
        defaults = RL_Config.DEFAULT_RIGHTWARD_REWARD_PARAMS
        recognised_params = (
            "x_velocity_factor",
            "angle_penalty_factor",
            "collision_penalty",
            "crash_penalty_multiplier"
        )
        for param in recognised_params:
            try:
                setattr(self, param, float(kwargs.get(param, defaults[param])))
            except (ValueError, TypeError):
                raise logger.fatal(f"{param} must be a float", exc_info=True)
        extra_params = set(kwargs) - set(recognised_params)
        for param in extra_params:
            logger.warning(f"Unrecognized parameter: {param}")

    def get_reward(self, env, done: bool) -> float:
        reward = 0.0

        # Penalize crash
        if done:
            if env.crash_state:
                reward -= env.collision_impulse * self.crash_penalty_multiplier
            logger.debug(f"Final reward: {reward:.2f}")
            return float(reward)

        # Reward rightward motion and heading angle towards right
        x_velocity = env.lander_velocity[0]
        angle_error = abs((env.lander_angle - np.pi / 2) % np.pi)
        reward += (
            x_velocity * self.x_velocity_factor - angle_error * self.angle_penalty_factor
        ) * Config.RENDER_TIME_STEP

        # Penalize collision
        if env.collision_state:
            reward -= self.collision_penalty * Config.RENDER_TIME_STEP

        return float(reward)
