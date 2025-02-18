import numpy as np
from lander_learner.utils.config import Config
import logging

logger = logging.getLogger(__name__)


def default_reward(env, done):
    """
    Default reward function:
      - Rewards flight to the right (using x-velocity).
      - Penalizes deviation from π/2 orientation (lander up towards right).
      - Penalizes collisions.
    """
    reward = 0.0

    # Penalize crash
    if done:
        if env.crash_state:
            reward -= env.collision_impulse * 1.0
        logger.debug(f"Final reward: {reward:.2f}")
        return float(reward)

    # Reward rightward travel and heading angle towards right
    x_velocity = env.lander_velocity[0]
    angle_penalty = abs((env.lander_angle - np.pi / 2) % np.pi)
    reward += (x_velocity * 10 - angle_penalty * 1.0) * Config.FRAME_TIME_STEP

    # Penalize collision
    if env.collision_state:
        reward -= 5.0 * Config.FRAME_TIME_STEP

    return float(reward)


def soft_landing_reward(env, done):
    """
    Sparse reward function:
      - Returns a penalty if a collision occurs.
      - Returns a positive reward only if the lander touches the ground softly (no collision).
      - Otherwise, no reward.
    """
    reward = 0.0

    # Penalize crash and reward soft landing in target zone
    if done:
        if env.crash_state:
            reward -= env.collision_impulse * 1.0
        elif env.idle_state:
            if (
                env.target_position[0] - env.target_zone_width / 2
                <= env.lander_position[0]
                <= env.target_position[0] + env.target_zone_width / 2
                and env.target_position[1] - env.target_zone_height / 2
                <= env.lander_position[1]
                <= env.target_position[1] + env.target_zone_height / 2
            ):
                reward += 20.0 * (Config.MAX_EPISODE_DURATION - env.elapsed_time)
            else:
                reward -= 2.0 * (Config.MAX_EPISODE_DURATION - env.elapsed_time)
        elif env.time_limit_reached:
            pass
        else:
            logger.warning("Unrecognised termination condition. No reward assigned.")
        logger.debug(f"Final reward: {reward:.2f}")
        return float(reward)

    # Reward travel toward target position
    try:
        vector_to_target = env.target_position - env.lander_position
    except NameError as e:
        raise ValueError(
            "Target position must be defined in environment to use soft landing reward. "
            "Set target_zone_mode to True when creating environment. ",
            e,
        )
    distance_to_target = np.linalg.norm(vector_to_target)
    reward += 2.0 * np.dot(env.lander_velocity, vector_to_target) / distance_to_target * Config.FRAME_TIME_STEP

    # Encourage being upright and moving slowly near the target
    angle_penalty = abs(((env.lander_angle + np.pi) % (2 * np.pi)) - np.pi) - np.pi / 4
    velocity_penalty = np.linalg.norm(env.lander_velocity) - 1.0
    time_penalty = 1.0
    reward -= (
        (angle_penalty * 1.0 + velocity_penalty * 2.0 + time_penalty * 1.0)
        * (5 / np.clip(distance_to_target, 2, np.inf))
        * Config.FRAME_TIME_STEP
    )

    # Penalize collision
    if env.collision_state:
        if distance_to_target < env.target_zone_width / 2:
            reward += 10.0 * Config.FRAME_TIME_STEP
        else:
            reward -= 5.0 * Config.FRAME_TIME_STEP

    return float(reward)


def get_reward_function(name: str):
    """
    Given a reward function name, return the corresponding function.
    Defaults to the `default_reward` if name is not recognized.
    """
    mapping = {
        "rightward": default_reward,
        "soft_landing": soft_landing_reward,
    }
    return mapping.get(name, default_reward)
