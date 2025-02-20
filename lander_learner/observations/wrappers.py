import numpy as np
import logging
from lander_learner.observations.base_observation import BaseObservation

logger = logging.getLogger(__name__)


class ObservationWrapper(BaseObservation):
    """Base class for observation wrappers.

    This class wraps an existing observation generator and can modify or extend its output.

    Attributes:
        observation (BaseObservation): The underlying observation generator.
        observation_size (int): The observation size as defined by the wrapped generator.
    """

    def __init__(self, observation: BaseObservation):
        """Initializes the ObservationWrapper.

        Args:
            observation (BaseObservation): The observation generator to wrap.
        """
        self.observation = observation
        self.observation_size = observation.observation_size

    def get_observation(self, env):
        """Generates an observation by delegating to the wrapped observation generator.

        Args:
            env: The environment instance.

        Returns:
            numpy.ndarray: The observation vector produced by the underlying generator.
        """
        return self.observation.get_observation(env)


class NoiseObservationWrapper(ObservationWrapper):
    """An observation wrapper that adds Gaussian noise to observations.

    The wrapper can use either a vector of noise variances (which will be converted into a diagonal covariance matrix)
    or a full covariance matrix for adding noise to the observation.

    Keyword Args:
        noise_variance (array-like): Variance for each observation dimension.
        noise_covariance (array-like): A full covariance matrix to sample noise from.
            Takes precedence over noise_variance.
    """

    def __init__(self, observation: BaseObservation, **kwargs):
        """Initializes the NoiseObservationWrapper.

        Args:
            observation (BaseObservation): The base observation generator to wrap.
            **kwargs: Additional keyword arguments for noise parameters. Recognized keys:
                - noise_variance: Variance for each observation dimension.
                - noise_covariance: A full covariance matrix for noise sampling.
        """
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
        """Generates a noisy observation by adding Gaussian noise to the base observation.

        The noise is sampled from a multivariate normal distribution with mean zero and the specified covariance.

        Args:
            env: The environment instance.

        Returns:
            numpy.ndarray: The noisy observation vector.

        Raises:
            ValueError: If the noise covariance matrix dimensions do not match the observation dimension.
        """
        obs = self.observation.get_observation(env)
        if self.noise_covariance is not None:
            n = obs.shape[0]
            if self.noise_covariance.shape != (n, n):
                raise ValueError("The provided noise covariance matrix shape does not match the observation dimension.")
            noise = np.random.multivariate_normal(np.zeros(n), self.noise_covariance)
            return obs + noise
        else:
            return obs
