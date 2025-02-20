import numpy as np
from lander_learner.rewards.base_reward import BaseReward
from lander_learner.utils.config import Config, RL_Config
import logging

logger = logging.getLogger(__name__)


class SoftLandingReward(BaseReward):
    """Computes a reward that promotes a soft landing near a target zone.

    The reward function applies different bonuses and penalties depending on:
      - Whether the lander touches down within the target zone.
      - How close the lander idles to the target.
      - The travel towards the target.
      - The orientation and velocity near the target.
      - Collision penalties.

    Attributes:
        on_target_touch_down_bonus (float): Bonus for a soft touchdown within the target zone.
        off_target_touch_down_penalty (float): Penalty for touchdown outside the target zone.
        on_target_idle_bonus (float): Bonus for idling within the target zone.
        off_target_idle_penalty (float): Penalty for idling outside the target zone.
        crash_penalty_multiplier (float): Multiplier for collision impulse penalty upon termination.
        time_penalty_factor (float): Factor for penalizing time taken.
        travel_reward_factor (float): Factor for rewarding travel toward the target.
        near_target_off_angle_penalty (float): Penalty for deviation from the target angle.
        near_target_high_velocity_penalty (float): Penalty for high velocity near the target.
        near_target_high_velocity_cut_off (float): Cutoff for the high velocity penalty.
        near_target_unit_dist (float): Unit distance for near-target calculations.
        near_target_max_multiplier (float): Maximum multiplier for near-target calculations.
        near_target_passive_bonus (float): Bonus for passive behavior near the target.
    """

    def __init__(self, **kwargs):
        """Initializes SoftLandingReward with configurable parameters.

        Args:
            **kwargs: Optional keyword arguments to override default parameters. Recognized keys:
                - on_target_touch_down_bonus (float)
                - off_target_touch_down_penalty (float)
                - on_target_idle_bonus (float)
                - off_target_idle_penalty (float)
                - crash_penalty_multiplier (float)
                - time_penalty_factor (float)
                - travel_reward_factor (float)
                - near_target_off_angle_penalty (float)
                - near_target_high_velocity_penalty (float)
                - near_target_high_velocity_cut_off (float)
                - near_target_unit_dist (float)
                - near_target_max_multiplier (float)
                - near_target_passive_bonus (float)

            Defaults are taken from RL_Config.DEFAULT_SOFT_LANDING_REWARD_PARAMS.
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
        """Computes the soft landing reward based on the environment state.

        The computation includes:
          - A bonus or penalty on touchdown depending on target proximity.
          - A reward for travel towards the target.
          - A penalty based on the lander's orientation and velocity near the target.
          - A collision penalty if the lander is in collision.

        Args:
            env: The environment instance containing state variables.
            done (bool): Flag indicating whether the episode is terminated.

        Returns:
            float: The computed soft landing reward.
        """
        reward = 0.0

        vector_to_target = env.target_position - env.lander_position
        distance_to_target = np.linalg.norm(vector_to_target)

        # If the episode is terminated, apply touchdown rewards or penalties.
        if done:
            if env.crash_state:
                reward -= (self.crash_penalty_multiplier * env.collision_impulse +
                           self.time_penalty_factor * (Config.MAX_EPISODE_DURATION - env.elapsed_time))
            elif env.idle_state:
                reward += (
                    self.on_target_idle_bonus
                    - (self.on_target_idle_bonus + self.off_target_idle_penalty)
                    * np.clip(distance_to_target / env.target_zone_width, 0.0, 1.0)
                ) * (Config.MAX_EPISODE_DURATION - env.elapsed_time)
            elif env.time_limit_reached:
                pass
            else:
                logger.warning("Unrecognised termination condition. No reward assigned.")
            logger.debug(f"Final reward: {reward:.2f}")
            return float(reward)

        # Reward travel toward the target.
        reward += (
            self.travel_reward_factor
            * np.dot(env.lander_velocity, vector_to_target)
            / distance_to_target
            - self.time_penalty_factor
        ) * Config.FRAME_TIME_STEP

        # Penalize deviations in orientation and excessive velocity near the target.
        angle_penalty = abs(((env.lander_angle + np.pi) % (2 * np.pi)) - np.pi) / np.pi
        velocity_penalty = np.linalg.norm(np.clip(env.lander_velocity, 1.0, None) - 1.0)
        reward -= (
            (self.near_target_off_angle_penalty * angle_penalty +
             self.near_target_high_velocity_penalty * velocity_penalty +
             self.near_target_passive_bonus)
            * (self.near_target_unit_dist /
               np.clip(distance_to_target, self.near_target_unit_dist / self.near_target_max_multiplier, np.inf))
            * Config.FRAME_TIME_STEP
        )

        # Penalize collision if it occurs near the target.
        if env.collision_state:
            reward += (
                self.on_target_touch_down_bonus
                - (self.on_target_touch_down_bonus + self.off_target_touch_down_penalty)
                * np.clip(distance_to_target / env.target_zone_width, 0.0, 1.0)
            ) * Config.FRAME_TIME_STEP

        return float(reward)
