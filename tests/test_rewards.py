import numpy as np
import pytest  # noqa: F401
from lander_learner.rewards import get_reward_class
from lander_learner.utils.config import Config


# A dummy environment class for reward testing.
class DummyEnv:
    def __init__(self):
        self.lander_position = np.array([0.0, 10.0], dtype=np.float32)
        self.lander_velocity = np.array([1.0, 0.0], dtype=np.float32)
        self.lander_angle = np.pi / 2  # upright
        self.lander_angular_velocity = 0.0
        self.fuel_remaining = Config.INITIAL_FUEL
        self.collision_state = False
        self.collision_impulse = 0.0
        self.elapsed_time = 5.0
        self.target_position = np.array([10.0, 0.0], dtype=np.float32)
        self.target_zone_width = 10.0
        self.target_zone_height = 5.0
        self.crash_state = False
        self.idle_state = False
        self.time_limit_reached = False


def test_default_reward_flight():
    env = DummyEnv()
    default_reward = get_reward_class("default")
    reward = default_reward.get_reward(env, done=False)
    # When not done, reward is computed from velocity and angle.
    assert isinstance(reward, float)


def test_default_reward_done_crash():
    env = DummyEnv()
    env.crash_state = True
    env.collision_impulse = 30.0
    default_reward = get_reward_class("default")
    reward = default_reward.get_reward(env, done=True)
    # Expect negative reward for a crash.
    assert reward < 0


def test_soft_landing_reward_done_landing_on_target():
    env = DummyEnv()
    env.idle_state = True
    env.lander_position = env.target_position.copy()
    soft_landing_reward = get_reward_class("soft_landing")
    reward = soft_landing_reward.get_reward(env, done=True)
    # For a soft landing on target, we expect a positive reward.
    assert reward > 0


def test_soft_landing_reward_done_landing_off_target():
    env = DummyEnv()
    env.idle_state = True
    # Move the lander off target zone by shifting its x position far from the target.
    env.lander_position[0] = env.target_position[0] + 2.0 * env.target_zone_width
    soft_landing_reward = get_reward_class("soft_landing")
    reward = soft_landing_reward.get_reward(env, done=True)
    # For a landing off the target zone, reward should be negative.
    assert reward < 0


def test_soft_landing_reward_done_no_landing():
    env = DummyEnv()
    env.time_limit_reached = True
    soft_landing_reward = get_reward_class("soft_landing")
    reward = soft_landing_reward.get_reward(env, done=True)
    # Reaching time limit, landing did not occur, so reward should be zero or negative.
    assert reward <= 0
