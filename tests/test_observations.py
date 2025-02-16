import numpy as np
import pytest
from lander_learner.utils.observations import default_observation, target_landing_observation

# Dummy environment for testing observation functions.
class DummyEnv:
    def __init__(self):
        self.lander_position = np.array([0.0, 10.0], dtype=np.float32)
        self.lander_velocity = np.array([1.0, 0.5], dtype=np.float32)
        self.lander_angle = np.pi / 2  # upright
        self.lander_angular_velocity = 0.0
        self.fuel_remaining = 100.0
        self.collision_state = False
        # For target-related observations:
        self.target_position = np.array([5.0, 0.0], dtype=np.float32)
        self.target_zone_width = 10.0
        self.target_zone_height = 5.0

def test_default_observation_shape():
    env = DummyEnv()
    obs = default_observation(env)
    assert isinstance(obs, np.ndarray)
    # Default observation should be 8-dimensional.
    assert obs.shape[0] == 8

def test_target_landing_observation_shape():
    env = DummyEnv()
    obs = target_landing_observation(env)
    # Target landing observation should be 13-dimensional (8 + 5 extra values).
    assert obs.shape[0] == 13
