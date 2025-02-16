from abc import ABC, abstractmethod


class BaseReward(ABC):
    def __init__(self, **kwargs):
        """
        Accepts parameter overrides via kwargs.
        """
        pass

    @abstractmethod
    def get_reward(self, env, done: bool) -> float:
        """
        Calculate and return the reward given the environment and termination flag.
        """
        pass
