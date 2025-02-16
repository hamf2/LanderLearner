from abc import ABC, abstractmethod


class BaseObservation(ABC):
    def __init__(self, **kwargs):
        """
        Optionally accept parameters and set the observation size.
        """
        self.observation_size = None

    @abstractmethod
    def get_observation(self, env):
        """
        Given the environment state, return an observation vector.
        """
        pass
