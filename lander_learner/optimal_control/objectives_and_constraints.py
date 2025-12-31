"""Example objectives and constraints for the modular lander trajectory optimizer.

This module defines small, composable objective and constraint classes
used by :class:`TrajectoryOptimizer`. Objectives return CasADi expressions
that the optimizer sums into a single objective; constraints apply
their conditions directly to the provided ``opti`` instance.
"""

from __future__ import annotations

from typing import Callable, Optional

import casadi as ca


class TrajectoryObjective:
    """Base class for trajectory objectives.

    Subclasses should implement :meth:`apply` and return a CasADi expression
    representing the contribution of the objective to the overall cost.
    """

    def apply(
        self,
        opti: ca.Opti,
        X: ca.MX,
        U: ca.MX,
        stage_duration_fn: Callable[[ca.MX, ca.MX, int], ca.MX],
    ) -> Optional[ca.MX]:
        """Return a CasADi expression representing this objective term.

        Args:
            opti: CasADi Opti builder instance (provided for convenience).
            X: State decision variable matrix with shape (state_size, N+1).
            U: Control decision variable matrix with shape (control_size, N).
            stage_duration_fn: Callable that returns the duration for a stage
                when needed by objective formulations.

        Returns:
            A CasADi expression (MX) representing the objective contribution,
            or ``None`` if the objective has no numeric expression to add.
        """
        raise NotImplementedError


class TrajectoryConstraint:
    """Base class for trajectory constraints.

    Subclasses implement :meth:`apply` and should add constraints to the
    provided ``opti`` object using ``opti.subject_to``.
    """

    def apply(self, opti: ca.Opti, X: ca.MX, U: ca.MX, stage_duration_fn: Callable[[ca.MX, ca.MX, int], ca.MX]) -> None:
        """Apply constraints to the Opti problem.

        Args:
            opti: CasADi Opti builder to receive constraints.
            X: State decision variable matrix.
            U: Control decision variable matrix.
            stage_duration_fn: Callable returning stage durations.
        """
        raise NotImplementedError


class MinimizeFinalDistanceObjective(TrajectoryObjective):
    def __init__(self, target_xy):
        self.target_xy = target_xy

    def apply(self, opti, X, U, stage_duration_fn):
        final_pos = X[0:2, -1]
        err = final_pos - ca.vertcat(*self.target_xy)
        return ca.sumsqr(err)


class MinimizeDistanceObjective(TrajectoryObjective):
    """Minimize summed position error across all stages.

    This objective computes \\sum_k ||p_k - p^*||^2 where p_k are the
    (x,y) components of the state at each timestep and p^* is the target.
    """

    def __init__(self, target_xy: tuple[float, float]):
        self.target_xy = target_xy

    def apply(
        self, opti: ca.Opti, X: ca.MX, U: ca.MX, stage_duration_fn: Callable[[ca.MX, ca.MX, int], ca.MX]
    ) -> ca.MX:
        """Return the sum-squared position error across the horizon.

        Args and semantics are the same as :meth:`TrajectoryObjective.apply`.
        """
        target_vec = ca.vertcat(*self.target_xy)
        # replicate target across horizon length
        N = X.shape[1]
        target_mat = ca.repmat(target_vec, 1, N)
        pos_err = X[0:2, :] - target_mat
        return ca.sumsqr(pos_err)


class MinimizeControlEffortObjective(TrajectoryObjective):
    """Quadratic penalty on control effort: weight * sum(U^2).

    The objective returns a CasADi expression that penalizes control
    magnitudes across the horizon.
    """

    def __init__(self, weight: float = 1e-2):
        self.weight = weight

    def apply(
        self, opti: ca.Opti, X: ca.MX, U: ca.MX, stage_duration_fn: Callable[[ca.MX, ca.MX, int], ca.MX]
    ) -> ca.MX:
        return self.weight * ca.sumsqr(U)


class InitialStateConstraint(TrajectoryConstraint):
    """Enforce the initial state equality X[:,0] == initial_state."""

    def __init__(self, initial_state: list | tuple | ca.MX):
        self.initial_state = initial_state

    def apply(self, opti: ca.Opti, X: ca.MX, U: ca.MX, stage_duration_fn: Callable[[ca.MX, ca.MX, int], ca.MX]) -> None:
        opti.subject_to(X[:, 0] == self.initial_state)


class FinalVelocityConstraint(TrajectoryConstraint):
    """Constraint that enforces small final horizontal and vertical speeds.

    The constraint uses `opti.bounded` to provide bounded expressions with
    well-behaved derivatives.
    """

    def __init__(self, vxy_tol: float = 1e-2):
        self.vxy_tol = vxy_tol

    def apply(self, opti: ca.Opti, X: ca.MX, U: ca.MX, stage_duration_fn: Callable[[ca.MX, ca.MX, int], ca.MX]) -> None:
        # Use opti.bounded for smooth gradients
        vx_final = X[2, -1]
        vy_final = X[3, -1]

        opti.subject_to(opti.bounded(-self.vxy_tol, vx_final, self.vxy_tol))
        opti.subject_to(opti.bounded(-self.vxy_tol, vy_final, self.vxy_tol))


class ControlBoundsConstraint(TrajectoryConstraint):
    """Simple box constraints on the control inputs across the horizon."""

    def __init__(self, u_min: float, u_max: float):
        self.u_min = u_min
        self.u_max = u_max

    def apply(self, opti: ca.Opti, X: ca.MX, U: ca.MX, stage_duration_fn: Callable[[ca.MX, ca.MX, int], ca.MX]) -> None:
        opti.subject_to(opti.bounded(self.u_min, U, self.u_max))


class FuelNonnegativeConstraint(TrajectoryConstraint):
    """Ensure the fuel state component remains non-negative at all nodes."""

    def apply(self, opti: ca.Opti, X: ca.MX, U: ca.MX, stage_duration_fn: Callable[[ca.MX, ca.MX, int], ca.MX]) -> None:
        opti.subject_to(X[6, :] >= 0.0)
