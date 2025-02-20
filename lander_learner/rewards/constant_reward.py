from .base_reward import BaseReward


class ConstantReward(BaseReward):
    """A reward that always returns a fixed constant value.

    Attributes:
        value (float): The constant reward value.
    """

    def __init__(self, value: float):
        """Initializes a ConstantReward instance.

        Args:
            value (float): The constant reward value.
        """
        self.value = value

    def get_reward(self, env, done: bool) -> float:
        """Returns the constant reward value.

        Args:
            env: The environment instance (unused).
            done (bool): Termination flag (unused).

        Returns:
            float: The constant reward value.
        """
        return self.value
