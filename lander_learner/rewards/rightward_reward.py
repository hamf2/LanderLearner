import numpy as np
from lander_learner.rewards.base_reward import BaseReward
from lander_learner.utils.config import Config, RL_Config
import logging

logger = logging.getLogger(__name__)


class RightwardReward(BaseReward):
    """Computes a reward that emphasizes rightward motion and proper orientation.

    This reward function promotes rightward velocity, penalizes deviations from an ideal heading,
    and applies a penalty during collisions.

    Attributes:
        x_velocity_factor (float): Factor for rewarding rightward velocity.
        angle_penalty_factor (float): Factor for penalizing deviation from π/2.
        collision_penalty (float): Penalty applied per time step when a collision is detected.
        crash_penalty_multiplier (float): Multiplier for collision impulse penalty upon termination.
    """

    def __init__(self, **kwargs):
        """Initializes RightwardReward with configurable parameters.

        Args:
            **kwargs: Optional keyword arguments to override default parameters. Recognized keys:
                - x_velocity_factor (float)
                - angle_penalty_factor (float)
                - collision_penalty (float)
                - crash_penalty_multiplier (float)

                Defaults are taken from RL_Config.DEFAULT_RIGHTWARD_REWARD_PARAMS.
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
        """Computes the rightward reward based on the environment state.

        If the episode is terminated and a crash occurred, a penalty based on the collision
        impulse is applied. Otherwise, the reward promotes rightward motion and penalizes deviation
        from an ideal angle, with an additional penalty during collisions.

        Args:
            env: The environment instance.
            done (bool): Flag indicating whether the episode is terminated.

        Returns:
            float: The computed reward.
        """
        reward = 0.0
        if done:
            if env.crash_state:
                reward -= env.collision_impulse * self.crash_penalty_multiplier
            logger.debug(f"Final reward: {reward:.2f}")
            return float(reward)

        x_velocity = env.lander_velocity[0]
        angle_error = abs((env.lander_angle - np.pi / 2) % np.pi)
        reward += (
            x_velocity * self.x_velocity_factor - angle_error * self.angle_penalty_factor
        ) * Config.FRAME_TIME_STEP

        if env.collision_state:
            reward -= self.collision_penalty * Config.FRAME_TIME_STEP

        return float(reward)
