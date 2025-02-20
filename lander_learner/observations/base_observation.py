from abc import ABC, abstractmethod


class BaseObservation(ABC):
    """Abstract base class for observation generators.

    This class defines the interface for generating observation vectors from an environment.
    Subclasses must override the `get_observation` method and set the `observation_size` attribute.

    Attributes:
        observation_size (int or None): The dimensionality of the observation vector. Should be set by subclasses.
    """

    def __init__(self, **kwargs):
        """Initializes the BaseObservation.

        Optionally accepts parameters to customize the observation; however, the default implementation
        only sets `observation_size` to None.

        Args:
            **kwargs: Arbitrary keyword arguments (not used by default).
        """
        self.observation_size = None

    @abstractmethod
    def get_observation(self, env):
        """Generates and returns an observation vector from the given environment.

        Args:
            env: The environment instance from which to generate the observation.

        Returns:
            numpy.ndarray: The observation vector.
        """
        pass
