import numpy as np
import pytest  # noqa: F401
from lander_learner.observations import get_observation_class
from lander_learner.observations.wrappers import NoiseObservationWrapper


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
    # Get the default observation class and instantiate it.
    obs_class = get_observation_class("default")
    obs = obs_class.get_observation(env)
    assert isinstance(obs, np.ndarray)
    # Default observation is expected to be 8-dimensional.
    assert obs.shape[0] == 8


def test_target_landing_observation_shape():
    env = DummyEnv()
    # Get the target landing observation class and instantiate it.
    obs_class = get_observation_class("target")
    obs = obs_class.get_observation(env)
    # Target landing observation should be 13-dimensional (8 base + 5 extra values).
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] == 13


def test_noise_observation_wrapper_zero_variance():
    env = DummyEnv()
    # Use a default observation with noise set to zero so that outputs remain unchanged.
    base_obs = get_observation_class("default")
    noise_obs = NoiseObservationWrapper(base_obs, noise_variance=[0.0] * base_obs.observation_size)
    base_output = base_obs.get_observation(env)
    noisy_output = noise_obs.get_observation(env)
    np.testing.assert_allclose(noisy_output, base_output, atol=1e-5)


def test_noise_observation_wrapper_nonzero_variance():
    env = DummyEnv()
    # Use a default observation with nonzero noise.
    base_obs = get_observation_class("default")
    noise_obs = NoiseObservationWrapper(base_obs, noise_variance=[1.0] * base_obs.observation_size)
    base_output = base_obs.get_observation(env)
    noisy_output = noise_obs.get_observation(env)
    # Verify the observation shape remains the same.
    assert noisy_output.shape == base_output.shape
    # With nonzero noise, the outputs should not be almost equal.
    assert not np.allclose(noisy_output, base_output, atol=1e-6), (
        "Noise observation should not be equal to the base observation "
        "(random failure at 2x10^-49)."
    )
    num_samples = 100
    differences = np.array([noise_obs.get_observation(env) - base_output for _ in range(num_samples)])
    sample_variance = np.var(differences.flatten())
    assert np.allclose(sample_variance, 1.0, rtol=0.3), (
        "Sample variance should be close to the noise variance. "
        "(random failure at 2x10^-9 probability)"
    )


def test_noise_observation_wrapper_selective_variance():
    env = DummyEnv()
    base_obs = get_observation_class("default")
    obs_size = base_obs.observation_size
    # Set nonzero noise only for the first half of the indices.
    selective_indices = list(range(obs_size // 2))
    noise_variance = [
        1.0 if i in selective_indices else 0.0 for i in range(obs_size)
    ]
    noise_obs = NoiseObservationWrapper(base_obs, noise_variance=noise_variance)
    base_output = base_obs.get_observation(env)
    noisy_output = noise_obs.get_observation(env)

    # Check that components with zero variance remain unchanged.
    assert np.allclose(
        noisy_output[len(selective_indices):],
        base_output[len(selective_indices):],
        atol=1e-6,
    ), "Zero noise indices were affected by noise."

    # Check that components with nonzero variance have noise added.
    differences = noisy_output[:len(selective_indices)] - base_output[:len(selective_indices)]
    # With nonzero noise, it is very unlikely that all differences are exactly zero.
    assert not np.allclose(
        differences, np.zeros_like(differences), atol=1e-6
    ), "Nonzero noise indices were not affected by noise. (random failure at 4x10^-25)"
