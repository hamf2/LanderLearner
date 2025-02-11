import numpy as np
from utils.config import Config

def default_reward(env, done):
    """
    Default reward function:
      - Rewards flight to the right (using x-velocity).
      - Penalizes deviation from π/2 orientation (lander up towards right).
      - Penalizes collisions.
    """
    x_velocity = env.lander_velocity[0]
    angle_penalty = abs((env.lander_angle - np.pi/2) % np.pi)
    reward = (x_velocity * 10 - angle_penalty * 1.0) * Config.RENDER_TIME_STEP

    if env.crash_state:
        reward -= 100.0
    elif env.collision_state:
        reward -= 5.0

    return reward

def soft_landing_reward(env, done):
    """
    Sparse reward function:
      - Returns a penalty if a collision occurs.
      - Returns a positive reward only if the lander touches the ground softly (no collision).
      - Otherwise, no reward.
    """
    if env.collision_state:
        return -100.0
    elif env.lander_position[1] <= 0.0 and not env.collision_state:
        return 100.0
    else:
        return 0.0

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