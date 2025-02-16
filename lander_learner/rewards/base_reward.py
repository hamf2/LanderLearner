from abc import ABC, abstractmethod
import operator


class BaseReward(ABC):
    def __init__(self, **kwargs):
        """
        Base class for all reward objects.
        Subclasses should override get_reward().

        Operator overloading is enabled:
          - r1 + r2 returns a CompositeReward computing r1.get_reward() + r2.get_reward()
          - r1 - r2, r1 * r2, r1 / r2, etc., are similarly supported.
          - Scalars are automatically wrapped in ConstantReward.
        """
        pass

    @abstractmethod
    def get_reward(self, env, done: bool) -> float:
        """
        Compute and return the reward given the environment state and termination flag.
        """
        pass

    # Operator overloads:
    def __add__(self, other):
        from .composite_reward import CompositeReward
        return CompositeReward(self, operator.add, other)

    def __radd__(self, other):
        from .composite_reward import CompositeReward
        return CompositeReward(other, operator.add, self)

    def __sub__(self, other):
        from .composite_reward import CompositeReward
        return CompositeReward(self, operator.sub, other)

    def __rsub__(self, other):
        from .composite_reward import CompositeReward
        return CompositeReward(other, operator.sub, self)

    def __mul__(self, other):
        from .composite_reward import CompositeReward
        return CompositeReward(self, operator.mul, other)

    def __rmul__(self, other):
        from .composite_reward import CompositeReward
        return CompositeReward(other, operator.mul, self)

    def __truediv__(self, other):
        from .composite_reward import CompositeReward
        return CompositeReward(self, operator.truediv, other)

    def __rtruediv__(self, other):
        from .composite_reward import CompositeReward
        return CompositeReward(other, operator.truediv, self)
