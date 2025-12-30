"""Example objectives and constraints for the modular lander trajectory optimizer."""

import casadi as ca


class TrajectoryObjective:
    def apply(self, opti, X, U, stage_duration_fn):
        """Return a CasADi expression representing this objective term.

        The returned expression will be summed with other objectives and
        passed to `opti.minimize(...)` once by the optimizer.
        """
        raise NotImplementedError


class TrajectoryConstraint:
    def apply(self, opti, X, U, stage_duration_fn):
        """Apply constraint to the opti problem. stage_duration_fn(X, U, k) gives duration for stage k."""
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

    This mirrors the temporal demo objective which guides the solver at
    every timestep rather than only at the terminal state.
    """

    def __init__(self, target_xy):
        self.target_xy = target_xy

    def apply(self, opti, X, U, stage_duration_fn):
        target_vec = ca.vertcat(*self.target_xy)
        # replicate target across horizon length
        N = X.shape[1]
        target_mat = ca.repmat(target_vec, 1, N)
        pos_err = X[0:2, :] - target_mat
        return ca.sumsqr(pos_err)


class MinimizeControlEffortObjective(TrajectoryObjective):
    def __init__(self, weight=1e-2):
        self.weight = weight

    def apply(self, opti, X, U, stage_duration_fn):
        return self.weight * ca.sumsqr(U)


class InitialStateConstraint(TrajectoryConstraint):
    def __init__(self, initial_state):
        self.initial_state = initial_state

    def apply(self, opti, X, U, stage_duration_fn):
        opti.subject_to(X[:, 0] == self.initial_state)


class FinalVelocityConstraint(TrajectoryConstraint):
    def __init__(self, vxy_tol=1e-2):
        self.vxy_tol = vxy_tol

    def apply(self, opti, X, U, stage_duration_fn):
        # Use opti.bounded for smooth gradients
        vx_final = X[2, -1]
        vy_final = X[3, -1]

        opti.subject_to(opti.bounded(-self.vxy_tol, vx_final, self.vxy_tol))
        opti.subject_to(opti.bounded(-self.vxy_tol, vy_final, self.vxy_tol))


class ControlBoundsConstraint(TrajectoryConstraint):
    def __init__(self, u_min, u_max):
        self.u_min = u_min
        self.u_max = u_max

    def apply(self, opti, X, U, stage_duration_fn):
        opti.subject_to(opti.bounded(self.u_min, U, self.u_max))


class FuelNonnegativeConstraint(TrajectoryConstraint):
    def apply(self, opti, X, U, stage_duration_fn):
        opti.subject_to(X[6, :] >= 0.0)
