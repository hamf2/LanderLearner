from .base_reward import BaseReward
from .constant_reward import ConstantReward


class CompositeReward(BaseReward):
    """A reward composed by applying a binary operation to two reward operands.

    The operands can be either instances of BaseReward or scalars (which are automatically
    wrapped in a ConstantReward).

    Attributes:
        left (BaseReward): The left operand.
        right (BaseReward): The right operand.
        op (callable): The binary operator used to combine the operands.
    """

    def __init__(self, left, op, right):
        """Initializes a CompositeReward instance.

        Args:
            left (BaseReward or scalar): The left operand.
            op (callable): A binary operator (e.g., operator.add) to apply.
            right (BaseReward or scalar): The right operand.
        """
        self.left = left if isinstance(left, BaseReward) else ConstantReward(left)
        self.right = right if isinstance(right, BaseReward) else ConstantReward(right)
        self.op = op

    def get_reward(self, env, done: bool) -> float:
        """Computes the composite reward by applying the operator to the operands' rewards.

        Args:
            env: The environment instance.
            done (bool): Flag indicating if the episode is terminated.

        Returns:
            float: The computed composite reward.
        """
        left_value = self.left.get_reward(env, done)
        right_value = self.right.get_reward(env, done)
        return self.op(left_value, right_value)
