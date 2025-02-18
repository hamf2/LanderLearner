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
            on_target_touch_down_bonus (float): Bonus reward for a soft landing within the target zone.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["on_target_touch_down_bonus"]
            off_target_touch_down_penalty (float): Penalty for touching down off target.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["off_target_touch_down_penalty"]
            on_target_idle_bonus (float): Bonus reward for idling within the target zone.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["on_target_idle_bonus"]
            off_target_idle_penalty (float): Penalty for idling off target.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["off_target_idle_penalty"]
            crash_penalty_multiplier (float): Multiplier for penalty based on collision impulse on termination.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["crash_penalty_multiplier"]
            time_penalty_factor (float): Factor for penalizing time taken.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["time_penalty_factor"]
            travel_reward_factor (float): Factor for rewarding travel towards the target.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["travel_reward_factor"]
            near_target_off_angle_penalty (float): Penalty for being off-angle near the target.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["near_target_off_angle_penalty"]
            near_target_high_velocity_penalty (float): Penalty for high velocity near the target.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["near_target_high_velocity_penalty"]
            near_target_unit_dist (float): Unit distance for near target calculations.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["near_target_unit_dist"]
            near_target_max_multiplier (float): Maximum multiplier for near target calculations.
                Default: RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS["near_target_max_multiplier"]
        """
        defaults = RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS
        recognized_params = (
            "on_target_touch_down_bonus",
            "off_target_touch_down_penalty",
            "on_target_idle_bonus",
            "off_target_idle_penalty",
            "crash_penalty_multiplier",
            "time_penalty_factor",
            "travel_reward_factor",
            "near_target_off_angle_penalty",
            "near_target_high_velocity_penalty",
            "near_target_high_velocity_cut_off",
            "near_target_unit_dist",
            "near_target_max_multiplier",
            "near_target_passive_bonus"
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

        vector_to_target = env.target_position - env.lander_position
        distance_to_target = np.linalg.norm(vector_to_target)

        # Penalize crash and reward soft landing in target zone
        if done:
            if env.crash_state:
                reward -= (self.crash_penalty_multiplier * env.collision_impulse
                           + self.time_penalty_factor * (Config.MAX_EPISODE_DURATION - env.elapsed_time))
            elif env.idle_state:
                reward += (
                    self.on_target_idle_bonus
                    - (self.on_target_idle_bonus + self.off_target_idle_penalty)
                    * np.clip(distance_to_target / env.target_zone_width, 0.0, 1.0)
                    ) * (Config.MAX_EPISODE_DURATION - env.elapsed_time)
                # in_target = (
                #     env.target_position[0] - env.target_zone_width / 2
                #     <= env.lander_position[0]
                #     <= env.target_position[0] + env.target_zone_width / 2
                #     and env.target_position[1] - env.target_zone_height / 2
                #     <= env.lander_position[1]
                #     <= env.target_position[1] + env.target_zone_height / 2
                # )
                # if in_target:
                #     reward += self.on_target_idle_bonus * (Config.MAX_EPISODE_DURATION - env.elapsed_time)
                # else:
                #     reward -= self.off_target_idle_penalty * (Config.MAX_EPISODE_DURATION - env.elapsed_time)
            elif env.time_limit_reached:
                pass
            else:
                logger.warning("Unrecognised termination condition. No reward assigned.")
            logger.debug(f"Final reward: {reward:.2f}")
            return float(reward)

        # Reward travel toward target position
        reward += (
            self.travel_reward_factor
            * np.dot(env.lander_velocity, vector_to_target)
            / distance_to_target
            - self.time_penalty_factor
            ) * Config.FRAME_TIME_STEP

        # Encourage being upright and moving slowly near the target
        angle_penalty = abs(((env.lander_angle + np.pi) % (2 * np.pi)) - np.pi) / np.pi
        velocity_penalty = np.linalg.norm(np.clip(env.lander_velocity, 1.0, None) - 1.0)
        reward -= (
            (self.near_target_off_angle_penalty * angle_penalty
             + self.near_target_high_velocity_penalty * velocity_penalty
             + self.near_target_passive_bonus)
            * (self.near_target_unit_dist
               / np.clip(distance_to_target, self.near_target_unit_dist / self.near_target_max_multiplier, np.inf))
            * Config.FRAME_TIME_STEP
        )

        # Penalize collision
        if env.collision_state:
            reward += (
                self.on_target_touch_down_bonus
                - (self.on_target_touch_down_bonus + self.off_target_touch_down_penalty)
                * np.clip(distance_to_target / env.target_zone_width, 0.0, 1.0)
                ) * Config.FRAME_TIME_STEP

        return float(reward)
