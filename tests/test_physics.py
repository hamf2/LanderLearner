import numpy as np
import pytest  # noqa: F401
from lander_learner.physics import PhysicsEngine
from lander_learner.utils.config import Config


# Create a dummy environment class for testing the physics engine.
class DummyEnv:
    def __init__(self):
        self.lander_position = np.array([0.0, 10.0], dtype=np.float32)
        self.lander_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.lander_angle = 0.0
        self.lander_angular_velocity = 0.0
        self.fuel_remaining = Config.INITIAL_FUEL
        self.collision_state = False
        self.collision_impulse = 0.0
        self.elapsed_time = 0.0


def test_physics_update():
    env = DummyEnv()
    physics_engine = PhysicsEngine()
    physics_engine.reset()
    initial_velocity = np.array(env.lander_velocity)
    initial_position = np.array(env.lander_position)
    # Apply max forces so that something happens.
    physics_engine.update(1.0, 1.0, env)
    # Check that the lander velocity has changed.
    assert not np.array_equal(env.lander_velocity, initial_velocity)
    # Apply max forces so that something happens.
    physics_engine.update(1.0, 1.0, env)
    # Check that the lander position has changed.
    assert not np.array_equal(env.lander_position, initial_position)
    # Fuel should have decreased.
    assert env.fuel_remaining < Config.INITIAL_FUEL


def test_physics_reset():
    physics_engine = PhysicsEngine()
    # Simulate a collision impulse
    physics_engine.collision_impulse = 50.0
    physics_engine.reset()
    # After reset, the collision impulse should be reset.
    assert physics_engine.collision_impulse == 0.0
