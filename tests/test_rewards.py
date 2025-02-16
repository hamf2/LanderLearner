import numpy as np
import pytest  # noqa: F401
from lander_learner.rewards import get_reward_class
from lander_learner.rewards.constant_reward import ConstantReward
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


def test_reward_addition():
    # Create two constant rewards with known values.
    reward1 = ConstantReward(5.0)
    reward2 = ConstantReward(3.0)
    # Using overloaded addition.
    composite = reward1 + reward2
    env = DummyEnv()
    # The composite reward should yield the sum of the two constant rewards.
    value = composite.get_reward(env, done=False)
    assert value == 8.0


def test_reward_multiplication_by_scalar():
    # Create a constant reward.
    reward = ConstantReward(4.0)
    scalar = 3.0
    # Multiply reward by a scalar using both left and right operations.
    product1 = reward * scalar
    product2 = scalar * reward
    env = DummyEnv()
    value1 = product1.get_reward(env, done=False)
    value2 = product2.get_reward(env, done=False)
    assert value1 == 12.0
    assert value2 == 12.0


def test_combined_rightward_and_soft_landing_scalars():
    # Test the addition of scalar multiples of a rightward reward and a soft_landing reward.
    # Scalars to use.
    a = 2.0
    b = 3.0
    # Create reward instances.
    rightward_reward = get_reward_class("rightward")
    soft_landing_reward = get_reward_class("soft_landing")
    # Compute scalar multiples and their sum.
    combined_reward = a * rightward_reward + b * soft_landing_reward

    env = DummyEnv()
    # Use a non-terminal scenario (done=False) for consistent reward computations.
    # Get individual rewards.
    reward_right = rightward_reward.get_reward(env, done=False)
    reward_soft = soft_landing_reward.get_reward(env, done=False)
    expected_value = a * reward_right + b * reward_soft
    composite_value = combined_reward.get_reward(env, done=False)
    # Allow a small numerical tolerance.
    assert abs(composite_value - expected_value) < 1e-6
