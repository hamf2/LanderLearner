from .base_reward import BaseReward
from .default_reward import DefaultReward
from .rightward_reward import RightwardReward
from .soft_landing_reward import SoftLandingReward


def get_reward_class(name: str, **kwargs) -> BaseReward:
    """
    Factory method to create a reward instance.

    Keyword arguments are passed to the reward constructor and used to override the default parameters.

    If an unknown name is provided, DefaultReward is used.
    """
    mapping = {
        "default": DefaultReward,
        "rightward": RightwardReward,
        "soft_landing": SoftLandingReward,
    }
    reward_cls = mapping.get(name, DefaultReward)
    return reward_cls(**kwargs)
