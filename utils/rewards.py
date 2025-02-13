import numpy as np
from utils.config import Config

def default_reward(env, done):
    """
    Default reward function:
      - Rewards flight to the right (using x-velocity).
      - Penalizes deviation from π/2 orientation (lander up towards right).
      - Penalizes collisions.
    """
    # Penalize crash
    if done:
        if env.crash_state:
            reward -= env.collision_impulse * 1.0
        print(f"Final reward: {reward:.2f}")
        return float(reward)
    
    # Reward rightward travel and heading angle towards right
    x_velocity = env.lander_velocity[0]
    angle_penalty = abs((env.lander_angle - np.pi/2) % np.pi)
    reward = (x_velocity * 10 - angle_penalty * 1.0) * Config.RENDER_TIME_STEP

    # Penalize collision
    if env.collision_state:
        reward -= 5.0 * Config.RENDER_TIME_STEP

    return float(reward)

def soft_landing_reward(env, done):
    """
    Sparse reward function:
      - Returns a penalty if a collision occurs.
      - Returns a positive reward only if the lander touches the ground softly (no collision).
      - Otherwise, no reward.
    """
    # Penalize crash and reward soft landing in target zone
    if done:
        if env.crash_state:
            reward -= env.collision_impulse * 1.0
        if env.landing_state:
            reward += 20.0 * (Config.MAX_EPISODE_DURATION - env.elapsed_time)
        print(f"Final reward: {reward:.2f}")
        return float(reward)
    
    # Reward travel toward target position
    try:
        vector_to_target = env.target_position - env.lander_position
    except NameError as e:
        raise ValueError("Target position must be defined in environment to use soft landing reward. "
                         "Set target_zone_mode to True when creating environment. ", e)
    distance_to_target = np.linalg.norm(vector_to_target)
    reward = np.dot(env.lander_velocity, vector_to_target) / distance_to_target * Config.RENDER_TIME_STEP

    # Encourage being upright and moving slowly near the target
    angle_penalty = abs(((env.lander_angle + np.pi) % (2 * np.pi)) - np.pi) - np.pi/2
    velocity_penalty = np.linalg.norm(env.lander_velocity) - 3.0
    reward -= (angle_penalty * 1.0 + velocity_penalty * 2.0) * (5/np.clip(distance_to_target, 2, np.inf)) * Config.RENDER_TIME_STEP

    # Penalize collision
    if env.collision_state:
        if distance_to_target < env.target_zone_width / 2:
            reward += 10.0 * Config.RENDER_TIME_STEP
        else:
            reward -= 5.0 * Config.RENDER_TIME_STEP
    
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