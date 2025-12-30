"""Integration schemes for transcription: Trapezoidal and Hermite-Simpson.

These schemes provide `create_variables` and `constrain_interval` hooks
used by the transcription strategies.
"""

from abc import ABC, abstractmethod


class IntegrationScheme(ABC):
    @abstractmethod
    def create_variables(self, opti, horizon, state_size, control_size):
        """Create any internal variables needed (e.g., midpoints).

        Returns an extras dictionary (may be empty) which the transcription
        will hold for use during interval constraint enforcement.
        """
        pass

    @abstractmethod
    def constrain_interval(self, opti, f_dynamics, X_k, X_next, U_k, U_next, h_step, extras=None, k=0):
        """Apply constraints to close the gap between k and k+1.

        f_dynamics: callable f(x,u) returning state derivative (CasADi expression)
        h_step: size of the interval (dt or ds)
        extras: dictionary returned by create_variables
        k: interval index (used by schemes with per-interval variables)
        """
        pass


class TrapezoidalScheme(IntegrationScheme):
    def create_variables(self, opti, horizon, state_size, control_size):
        return {}

    def constrain_interval(self, opti, f_dynamics, X_k, X_next, U_k, U_next, h_step, extras=None, k=0):
        f_k = f_dynamics(X_k, U_k)
        f_next = f_dynamics(X_next, U_next)
        opti.subject_to(X_next == X_k + 0.5 * h_step * (f_k + f_next))


class HermiteSimpsonScheme(IntegrationScheme):
    def create_variables(self, opti, horizon, state_size, control_size):
        X_mid = opti.variable(state_size, horizon)
        U_mid = opti.variable(control_size, horizon)
        return {"X_mid": X_mid, "U_mid": U_mid}

    def constrain_interval(self, opti, f_dynamics, X_k, X_next, U_k, U_next, h_step, extras=None, k=0):
        if extras is None:
            raise ValueError("HermiteSimpsonScheme requires extras created by create_variables")
        X_c = extras["X_mid"][:, k]
        U_c = extras["U_mid"][:, k]

        f_k = f_dynamics(X_k, U_k)
        f_next = f_dynamics(X_next, U_next)
        f_c = f_dynamics(X_c, U_c)

        opti.subject_to(X_next == X_k + (h_step / 6.0) * (f_k + 4 * f_c + f_next))
        opti.subject_to(X_c == 0.5 * (X_k + X_next) + (h_step / 8.0) * (f_k - f_next))
