import numpy as np
from lander_learner.utils.config import Config

def default_observation(env):
    """
    Default observation: returns a 12-dimensional vector consisting of:
      - Lander position (x, y)
      - Lander velocity (vx, vy)
      - Lander angle and angular velocity
      - Fuel remaining
      - Distance to target (dx, dy)
      - Left and right laser distances
      - Collision state (as float)
    
    This provides a basic set of information about the lander state for RL agents
    but no additional information on goals or targets.
    """
    angle = (env.lander_angle + np.pi) % (2 * np.pi) - np.pi
    # altitude = env.lander_position[1]
    # left_laser_distance = Config.LASER_RANGE if angle >= 0 or angle == -np.pi else np.clip(
    #     altitude / np.sin(angle), 0, Config.LASER_RANGE)
    # right_laser_distance = Config.LASER_RANGE if angle <= 0 else np.clip(
    #     altitude / np.sin(-angle), 0, Config.LASER_RANGE)

    observation = np.array([
        env.lander_position[0],
        env.lander_position[1],
        env.lander_velocity[0],
        env.lander_velocity[1],
        angle,
        env.lander_angular_velocity,
        env.fuel_remaining,
        float(env.collision_state)
    ], dtype=np.float32)
    return observation

def target_landing_observation(env):
    """
    Extended observation: returns a 17-dimensional vector consisting of:
      - Default observations (12)
      - Target zone relative position (x, y)
      - Target zone direction
      - Target zone width and height
      
    This provides a basic lander state information with additional details 
    on target position for RL agents.
    """
    base_obs = default_observation(env)
    try:
        additional = np.array([
            env.target_position[0] - env.lander_position[0],
            env.target_position[1] - env.lander_position[1],
            np.arctan2(env.target_position[1] - env.lander_position[1], env.target_position[0] - env.lander_position[0]),
            env.target_zone_width,
            env.target_zone_height
        ], dtype=np.float32)
    except NameError as e:
        raise ValueError("Target position must be defined in environment to use target landing observation. "
                         "Set target_zone_mode to True when creating environment. ", e)
    return np.concatenate([base_obs, additional])

def get_observation_function(name: str):
    """
    Return a tuple of the observation function corresponding to the 
    given name and the size of the associated observation vector.
    Defaults to `default_observation` if the name is not recognized.
    """
    mapping = {
        "default": (default_observation, 8),
        "landing_zone": (target_landing_observation, 13)
    }
    return mapping.get(name, default_observation)