import numpy as np
from lander_learner.rewards.base_reward import BaseReward
from lander_learner.utils.config import Config
from lander_learner.utils.rl_config import RL_Config
import logging

logger = logging.getLogger(__name__)


class SoftLandingReward(BaseReward):
    def __init__(self, **kwargs):
        """
        Initialize SoftLandingReward with configurable parameters.

        Possible keyword arguments:
            soft_landing_bonus (float): Bonus reward for a soft landing within the target zone.
                                        Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["soft_landing_bonus"]
            crash_penalty_multiplier (float): Multiplier for penalty based on collision impulse on termination.
                                              Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["
                                              crash_penalty_multiplier"]
            time_penalty_factor (float): Factor for penalizing time taken.
                                         Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["time_penalty_factor"]
            travel_reward_factor (float): Factor for rewarding travel towards the target.
                                          Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["travel_reward_factor"]
        """
        defaults = RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS
        recognized_params = (
            "soft_landing_bonus",
            "crash_penalty_multiplier",
            "time_penalty_factor",
            "travel_reward_factor"
        )
        for param in recognized_params:
            try:
                setattr(self, param, float(kwargs.get(param, defaults[param])))
            except (ValueError, TypeError):
                raise logger.fatal(f"{param} must be a float", exc_info=True)
        extra_params = set(kwargs) - set(recognized_params)
        for param in extra_params:
            logger.warning(f"Unrecognized parameter: {param}")

    def get_reward(self, env, done: bool) -> float:
        reward = 0.0

        # Penalize crash and reward soft landing in target zone
        if done:
            if env.crash_state:
                reward -= env.collision_impulse * self.crash_penalty_multiplier
            elif env.idle_state:
                in_target = (
                    env.target_position[0] - env.target_zone_width / 2
                    <= env.lander_position[0]
                    <= env.target_position[0] + env.target_zone_width / 2
                    and env.target_position[1] - env.target_zone_height / 2
                    <= env.lander_position[1]
                    <= env.target_position[1] + env.target_zone_height / 2
                )
                if in_target:
                    reward += self.soft_landing_bonus * (Config.MAX_EPISODE_DURATION - env.elapsed_time)
                else:
                    reward -= 2.0 * (Config.MAX_EPISODE_DURATION - env.elapsed_time)
            elif env.time_limit_reached:
                pass
            else:
                logger.warning("Unrecognised termination condition. No reward assigned.")
            logger.debug(f"Final reward: {reward:.2f}")
            return float(reward)

        # Reward travel toward target position
        vector_to_target = env.target_position - env.lander_position
        distance_to_target = np.linalg.norm(vector_to_target)
        reward += (
            self.travel_reward_factor
            * np.dot(env.lander_velocity, vector_to_target)
            / distance_to_target
            * Config.RENDER_TIME_STEP
        )

        # Encourage being upright and moving slowly near the target
        angle_penalty = abs(((env.lander_angle + np.pi) % (2 * np.pi)) - np.pi) - np.pi / 4
        velocity_penalty = np.linalg.norm(env.lander_velocity) - 1.0
        reward -= (
            (angle_penalty + 2.0 * velocity_penalty + self.time_penalty_factor)
            * (5 / np.clip(distance_to_target, 2, np.inf))
            * Config.RENDER_TIME_STEP
        )

        # Penalize collision
        if env.collision_state:
            if distance_to_target < env.target_zone_width / 2:
                reward += 10.0 * Config.RENDER_TIME_STEP
            else:
                reward -= 5.0 * Config.RENDER_TIME_STEP

        return float(reward)
