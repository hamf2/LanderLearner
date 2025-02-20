from abc import ABC, abstractmethod
import operator


class BaseReward(ABC):
    """Base class for all reward objects.

    Subclasses must implement the get_reward() method to compute the reward based on
    the environment state and termination condition.

    Operator overloading is enabled:
      - r1 + r2 returns a CompositeReward computing r1.get_reward() + r2.get_reward()
      - Similarly, subtraction, multiplication, and division are supported.
      - Scalars are automatically wrapped in a ConstantReward.
    """

    def __init__(self, **kwargs):
        """Initializes a BaseReward instance.

        Args:
            **kwargs: Arbitrary keyword arguments for subclass configuration.
        """
        pass

    @abstractmethod
    def get_reward(self, env, done: bool) -> float:
        """Computes and returns the reward.

        Args:
            env: The environment instance containing the state.
            done (bool): A flag indicating whether the episode is terminated.

        Returns:
            float: The computed reward.
        """
        pass

    def __add__(self, other):
        """Overloads the '+' operator to combine rewards.

        Args:
            other: A reward or scalar.

        Returns:
            CompositeReward: A reward representing the sum.
        """
        from .composite_reward import CompositeReward
        return CompositeReward(self, operator.add, other)

    def __radd__(self, other):
        """Overloads the reverse '+' operator.

        Args:
            other: A scalar or reward.

        Returns:
            CompositeReward: A reward representing the sum.
        """
        from .composite_reward import CompositeReward
        return CompositeReward(other, operator.add, self)

    def __sub__(self, other):
        """Overloads the '-' operator to subtract rewards.

        Args:
            other: A reward or scalar.

        Returns:
            CompositeReward: A reward representing the difference.
        """
        from .composite_reward import CompositeReward
        return CompositeReward(self, operator.sub, other)

    def __rsub__(self, other):
        """Overloads the reverse '-' operator.

        Args:
            other: A scalar or reward.

        Returns:
            CompositeReward: A reward representing the difference.
        """
        from .composite_reward import CompositeReward
        return CompositeReward(other, operator.sub, self)

    def __mul__(self, other):
        """Overloads the '*' operator to multiply rewards.

        Args:
            other: A reward or scalar.

        Returns:
            CompositeReward: A reward representing the product.
        """
        from .composite_reward import CompositeReward
        return CompositeReward(self, operator.mul, other)

    def __rmul__(self, other):
        """Overloads the reverse '*' operator.

        Args:
            other: A scalar or reward.

        Returns:
            CompositeReward: A reward representing the product.
        """
        from .composite_reward import CompositeReward
        return CompositeReward(other, operator.mul, self)

    def __truediv__(self, other):
        """Overloads the '/' operator to divide rewards.

        Args:
            other: A reward or scalar.

        Returns:
            CompositeReward: A reward representing the quotient.
        """
        from .composite_reward import CompositeReward
        return CompositeReward(self, operator.truediv, other)

    def __rtruediv__(self, other):
        """Overloads the reverse '/' operator.

        Args:
            other: A scalar or reward.

        Returns:
            CompositeReward: A reward representing the quotient.
        """
        from .composite_reward import CompositeReward
        return CompositeReward(other, operator.truediv, self)
