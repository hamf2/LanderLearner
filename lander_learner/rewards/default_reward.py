import numpy as np
from lander_learner.rewards.base_reward import BaseReward
from lander_learner.utils.config import Config
from lander_learner.utils.rl_config import RL_Config


class DefaultReward(BaseReward):
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
        defaults = RL_Config.DEFAULT_DEFAULT_REWARD_PARAMS
        self.x_velocity_factor = kwargs.get("x_velocity_factor", defaults["x_velocity_factor"])
        self.angle_penalty_factor = kwargs.get("angle_penalty_factor", defaults["angle_penalty_factor"])
        self.collision_penalty = kwargs.get("collision_penalty", defaults["collision_penalty"])
        self.crash_penalty_multiplier = kwargs.get("crash_penalty_multiplier", defaults["crash_penalty_multiplier"])

    def get_reward(self, env, done: bool) -> float:
        reward = 0.0
        if done:
            if env.crash_state:
                reward -= env.collision_impulse * self.crash_penalty_multiplier
            return float(reward)

        # Reward rightward motion and penalize deviation from π/2.
        x_velocity = env.lander_velocity[0]
        angle_error = abs((env.lander_angle - np.pi / 2) % np.pi)
        reward += (
            x_velocity * self.x_velocity_factor - angle_error * self.angle_penalty_factor
        ) * Config.RENDER_TIME_STEP

        if env.collision_state:
            reward -= self.collision_penalty * Config.RENDER_TIME_STEP

        return float(reward)
