from .base_reward import BaseReward
from .constant_reward import ConstantReward


class CompositeReward(BaseReward):
    """
    CompositeReward applies a binary operation on two reward operands.

    The operands can be either:
      - Instances of BaseReward
      - Scalars (automatically wrapped into ConstantReward)

    Parameters:
        left (BaseReward or scalar): The left operand.
        op (callable): A binary operator function (e.g. operator.add).
        right (BaseReward or scalar): The right operand.
    """
    def __init__(self, left, op, right):
        self.left = left if isinstance(left, BaseReward) else ConstantReward(left)
        self.right = right if isinstance(right, BaseReward) else ConstantReward(right)
        self.op = op

    def get_reward(self, env, done: bool) -> float:
        left_value = self.left.get_reward(env, done)
        right_value = self.right.get_reward(env, done)
        return self.op(left_value, right_value)
