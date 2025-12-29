import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from lander_learner.environment import LunarLanderEnv
from lander_learner.utils.config import Config
from lander_learner.levels.level_data import load_level_payload


# Use a dummy physics engine to avoid running a full pymunk simulation.
class DummyPhysicsEngine:
    def __init__(self):
        self.updated = False
        self.metadata = {"episode_time_limit": None, "initial_fuel": None}

    def update(self, left_thruster, right_thruster, env):
        self.updated = True
        # For testing, just nudge the position by a small amount
        env.lander_position += np.array([left_thruster, right_thruster], dtype=np.float32)
        # Simulate fuel consumption
        fuel_used = (abs(left_thruster + 1) + abs(right_thruster + 1)) * 0.1
        env.fuel_remaining = max(0.0, env.fuel_remaining - fuel_used)

    def reset(self, env=None):
        self.updated = False
        if env is not None:
            env.lander_position = np.array([0.0, 10.0], dtype=np.float32)

    def get_bounds(self):
        # Mirror the default half-plane bounds used by the production engine.
        return (float("-inf"), float("inf"), 0.0, float("inf"))

    def get_level_metadata(self):
        return self.metadata.copy()


@pytest.fixture
def env_default():
    # Create an environment with default reward/observation and no target zone.
    env = LunarLanderEnv(
        gui_enabled=False, reward_function="rightward", observation_function="default", target_zone=False
    )
    # Replace the real physics engine with our dummy.
    env.physics_engine = DummyPhysicsEngine()
    env.reset()
    return env


def test_gymnasium_env_specification_match(env_default):
    # Check that the environment meets the Gym API requirements.
    check_env(env_default)


def test_time_limit_reached(env_default):
    # Trigger done flag by setting elapsed_time to MAX_EPISODE_DURATION.
    env_default.elapsed_time = Config.MAX_EPISODE_DURATION
    obs, reward, done, truncated, info = env_default.step([0.0, 0.0])
    assert done is True


def test_preset_time_limit_override_applied():
    env = LunarLanderEnv(level_name="p2p001", target_zone=False)
    try:
        payload = load_level_payload("p2p001")
        expected_limit = (
            payload.get("episode_time_limit")
            or payload.get("time_limit_seconds")
            or payload.get("time_limit")
        )
        if expected_limit is None:
            expected_limit = Config.MAX_EPISODE_DURATION
        assert env.max_episode_duration == pytest.approx(float(expected_limit))
        expected_fuel = payload.get("initial_fuel") or payload.get("starting_fuel")
        if expected_fuel is None:
            expected_fuel = Config.INITIAL_FUEL
        assert env.initial_fuel == pytest.approx(float(expected_fuel))
        assert float(env.fuel_remaining) == pytest.approx(float(expected_fuel))
        env.reset()
        assert float(env.fuel_remaining) == pytest.approx(float(expected_fuel))
        env.elapsed_time = env.max_episode_duration
        obs, reward, done, truncated, info = env.step([0.0, 0.0])
        assert done is True
    finally:
        env.close()


def test_crash_due_to_impulse(env_default):
    # Trigger crash: collision state with impulse exceeding threshold.
    env_default.collision_state = True
    env_default.collision_impulse = Config.IMPULSE_THRESHOLD + 1.0
    obs, reward, done, truncated, info = env_default.step([0.0, 0.0])
    assert done is True
    assert env_default.crash_state is True


def test_crash_due_to_angle(env_default):
    # Trigger crash: collision state with upside-down lander (angle between 90 and 270 degrees).
    env_default.collision_state = True
    env_default.lander_angle = np.pi  # 180 degrees.
    env_default.collision_impulse = 0.0
    obs, reward, done, truncated, info = env_default.step([0.0, 0.0])
    assert done is True
    assert env_default.crash_state is True


def test_lander_below_ground(env_default):
    # Trigger termination: lander position below ground level.
    env_default.lander_position[1] = -1.0
    obs, reward, done, truncated, info = env_default.step([0.0, 0.0])
    assert done is True
    assert env_default.collision_state is True


def test_idle_timeout_leads_to_done(env_default):
    # Trigger idle timeout: set collision state and nearly zero velocity, then simulate multiple steps.
    env_default.collision_state = True
    env_default.lander_velocity = np.array([0.05, 0.05], dtype=np.float32)
    # Simulate enough steps to exceed the idle timeout.
    steps = int(Config.IDLE_TIMEOUT / Config.TIME_STEP) + 1
    done_flag = False
    for _ in range(steps):
        obs, reward, done, truncated, info = env_default.step([0.0, 0.0])
        if done:
            done_flag = True
            break
    assert done_flag is True


def test_fuel_depletion_does_not_immediately_end_episode(env_default):
    # Fuel is depleted: env should not terminate immediately; let it coast.
    env_default.fuel_remaining = 0.0
    obs, reward, done, truncated, info = env_default.step([0.0, 0.0])
    assert done is False


def test_normal_step_no_termination(env_default):
    # Under normal conditions, a step should not trigger termination.
    obs, reward, done, truncated, info = env_default.step([0.1, -0.1])
    assert done is False
    assert env_default.physics_engine.updated is True


def test_reset(env_default):
    # Test that the environment can be reset without errors.
    env_default.step([1.0, 1.0])
    env_default.step([1.0, 1.0])
    env_default.reset()
    assert env_default.physics_engine.updated is False
    assert env_default.elapsed_time == 0.0
    assert env_default.fuel_remaining == Config.INITIAL_FUEL
    assert env_default.collision_state is False
    assert env_default.collision_impulse == 0.0
    assert env_default.crash_state is False
    assert env_default.idle_state is False
    assert env_default.idle_timer == 0.0
    assert env_default.time_limit_reached is False
    assert env_default.target_zone is False
    assert env_default.gui_enabled is False
    assert env_default.observation_space.shape == (8,)
    assert env_default.action_space.shape == (2,)
    assert env_default.metadata == {"render_modes": []}


def test_lap_finish_line_metadata_drives_sync(monkeypatch):
    env = LunarLanderEnv(level_name="lap001", target_zone=False)
    try:
        metadata = env.get_level_metadata()
        finish_points = metadata.get("finish_line_points")
        assert finish_points is not None

        monkeypatch.setattr(
            env.physics_engine,
            "get_level_metadata",
            lambda: {"finish_line_points": finish_points},
        )
        env.physics_engine.level = None
        env.finish_line = None
        env._sync_finish_line()

        assert env.finish_line is not None
        left, right = env.finish_line
        expected_left = np.asarray(finish_points[0], dtype=np.float32)
        expected_right = np.asarray(finish_points[1], dtype=np.float32)
        assert np.allclose(left, expected_left)
        assert np.allclose(right, expected_right)
    finally:
        env.close()
