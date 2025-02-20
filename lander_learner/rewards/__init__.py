"""
Rewards Package

This package provides classes and functions for computing rewards in the Lunar Lander environment.
It defines a common interface via BaseReward and includes various implementations, such as:
  - DefaultReward: A reward function that promotes rightward motion and proper orientation.
  - RightwardReward: A reward that emphasizes rightward movement.
  - SoftLandingReward: A reward that encourages soft landings near a target zone.

Reward objects support operator overloading, allowing them to be composed arithmetically.
The factory function get_reward_class() allows for easy instantiation of reward functions based on a name.
"""
from .base_reward import BaseReward
from .default_reward import DefaultReward
from .rightward_reward import RightwardReward
from .soft_landing_reward import SoftLandingReward


def get_reward_class(name: str, **kwargs) -> BaseReward:
    """Factory method to create a reward instance based on its name.

    Args:
        name (str): The name of the reward function (e.g., "default", "rightward", "soft_landing").
        **kwargs: Additional keyword arguments for configuring the reward instance.

    Returns:
        BaseReward: An instance of the requested reward function.
    """
    mapping = {
        "default": DefaultReward,
        "rightward": RightwardReward,
        "soft_landing": SoftLandingReward,
    }
    reward_cls = mapping.get(name, DefaultReward)
    return reward_cls(**kwargs)
