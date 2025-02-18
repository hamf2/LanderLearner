import numpy as np
import logging
from lander_learner.observations.base_observation import BaseObservation

logger = logging.getLogger(__name__)


class ObservationWrapper(BaseObservation):
    """
    Base class for observation wrappers.
    Wraps an existing observation object and can modify its output.

    Parameters:
        observation (BaseObservation): The underlying observation generator to wrap.
    """
    def __init__(self, observation: BaseObservation):
        self.observation = observation
        self.observation_size = observation.observation_size

    def get_observation(self, env):
        return self.observation.get_observation(env)


class NoiseObservationWrapper(ObservationWrapper):
    """
    A wrapper that adds Gaussian noise to the observations.

    Keyword Arguments (via **kwargs):
      - noise_variance (array-like): A vector of variances for each observation dimension.
                                     Noise is added as independent Gaussian noise if no covariance is provided.
      - noise_covariance (array-like): A full covariance matrix to sample noise from a multivariate normal distribution.
                                     If provided, this takes precedence over noise_variance.

    Example usage:

        from lander_learner.observations.default_observation import DefaultObservation
        from lander_learner.observations.wrappers import NoiseObservationWrapper

        base_obs = DefaultObservation()
        # Add noise with a variance vector for an 8-dimensional observation:
        noisy_obs = NoiseObservationWrapper(base_obs, noise_variance=[0.1]*8)

        # Alternatively, using a full covariance matrix:
        cov = [[0.1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0.1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0.1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0.1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0.1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0.1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0.1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0.1]]
        noisy_obs = NoiseObservationWrapper(base_obs, noise_covariance=cov)
    """
    def __init__(self, observation: BaseObservation, **kwargs):
        super().__init__(observation)
        self.noise_variance = kwargs.get("noise_variance", None)
        self.noise_covariance = kwargs.get("noise_covariance", None)
        if self.noise_covariance is None and self.noise_variance is not None:
            # Convert the variance vector into a diagonal covariance matrix.
            self.noise_covariance = np.diag(self.noise_variance)
        elif self.noise_covariance is None and self.noise_variance is None:
            # If no noise parameters are provided, do not add noise.
            self.noise_covariance = None
            logger.warning("Noise parameters not provided. No noise will be added to observations.")

    def get_observation(self, env):
        obs = self.observation.get_observation(env)
        if self.noise_covariance is not None:
            n = obs.shape[0]
            if self.noise_covariance.shape != (n, n):
                raise ValueError("The provided noise covariance matrix shape does not match the observation dimension.")
            # Sample noise from a multivariate normal distribution with mean zero.
            noise = np.random.multivariate_normal(np.zeros(n), self.noise_covariance)
            return obs + noise
        else:
            return obs
