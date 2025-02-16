from .base_reward import BaseReward


class ConstantReward(BaseReward):
    """
    A constant reward that always returns a fixed scalar value.

    Parameters:
        value (float): The constant reward value.
    """
    def __init__(self, value: float):
        self.value = value

    def get_reward(self, env, done: bool) -> float:
        return self.value
