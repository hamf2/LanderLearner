import numpy as np
from lander_learner.rewards.base_reward import BaseReward
from lander_learner.utils.config import Config, RL_Config


class DefaultReward(BaseReward):
    """Computes a default reward based on rightward motion and orientation.

    The reward encourages rightward velocity and penalizes deviations from an upright orientation.
    Additional penalties are applied for collisions.

    Attributes:
        x_velocity_factor (float): Factor for rewarding rightward velocity.
        angle_penalty_factor (float): Factor for penalizing deviation from π/2.
        collision_penalty (float): Penalty applied per time step during collision.
        crash_penalty_multiplier (float): Multiplier for collision impulse penalty upon termination.
    """

    def __init__(self, **kwargs):
        """Initializes DefaultReward with configurable parameters.

        Args:
            **kwargs: Optional keyword arguments to override default parameters. Recognized keys:
                - x_velocity_factor (float)
                - angle_penalty_factor (float)
                - collision_penalty (float)
                - crash_penalty_multiplier (float)

                Defaults are taken from RL_Config.DEFAULT_DEFAULT_REWARD_PARAMS.
        """
        defaults = RL_Config.DEFAULT_DEFAULT_REWARD_PARAMS
        self.x_velocity_factor = kwargs.get("x_velocity_factor", defaults["x_velocity_factor"])
        self.angle_penalty_factor = kwargs.get("angle_penalty_factor", defaults["angle_penalty_factor"])
        self.collision_penalty = kwargs.get("collision_penalty", defaults["collision_penalty"])
        self.crash_penalty_multiplier = kwargs.get("crash_penalty_multiplier", defaults["crash_penalty_multiplier"])

    def get_reward(self, env, done: bool) -> float:
        """Computes the default reward based on the environment state.

        If the episode is terminated and a crash occurred, a penalty proportional to the collision
        impulse is applied. Otherwise, the reward promotes rightward motion and penalizes deviation
        from an ideal angle, as well as applying a penalty during collisions.

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
            return float(reward)

        # Reward rightward motion and penalize angle deviation.
        x_velocity = env.lander_velocity[0]
        angle_error = abs((env.lander_angle - np.pi / 2) % np.pi)
        reward += (
            x_velocity * self.x_velocity_factor - angle_error * self.angle_penalty_factor
        ) * Config.FRAME_TIME_STEP

        if env.collision_state:
            reward -= self.collision_penalty * Config.FRAME_TIME_STEP

        return float(reward)
