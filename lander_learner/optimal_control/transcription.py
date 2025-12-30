"""Modular transcription strategies for lander optimal control (time-stepping and space-stepping)."""

from abc import ABC, abstractmethod
import casadi as ca


class TranscriptionStrategy(ABC):
    @abstractmethod
    def initialize_variables(self, opti, horizon_len, state_size, control_size):
        """Creates the decision variables X and U with appropriate shapes."""
        pass

    @abstractmethod
    def apply_dynamics(self, opti, dynamics_model, X, U, **kwargs):
        """
        Applies the continuity constraints (the 'gap closing').
        """
        pass

    @abstractmethod
    def get_stage_duration(self, X, U, k, **kwargs):
        """
        Returns the time elapsed during step k.
        """
        pass


class TimeSteppingTranscription(TranscriptionStrategy):
    def __init__(self, dt=0.1, scheme=None, variable_dt=False, dt_min=1e-4, dt_max=10.0):
        self.dt = dt
        self.variable_dt = variable_dt
        self.dt_min = dt_min
        self.dt_max = dt_max
        from .integration_schemes import TrapezoidalScheme

        self.scheme = scheme if scheme is not None else TrapezoidalScheme()
        self.extras = None

    def initialize_variables(self, opti, horizon_len, state_size, control_size):
        X = opti.variable(state_size, horizon_len + 1)
        U = opti.variable(control_size, horizon_len)
        # variable timestep remains optional
        if self.variable_dt:
            dt_var = opti.variable(horizon_len)
            opti.subject_to(dt_var >= self.dt_min)
            opti.subject_to(dt_var <= self.dt_max)
            opti.set_initial(dt_var, self.dt)
        else:
            dt_var = None

        # create scheme-specific internal variables (e.g. midpoints)
        self.extras = self.scheme.create_variables(opti, horizon_len, state_size, control_size)
        return X, U, dt_var

    def apply_dynamics(self, opti, dynamics_model, X, U, dt_var=None):
        # define continuous-time derivative wrapper
        def time_dynamics(x, u):
            return dynamics_model.state_derivative(x, u)

        N = U.shape[1]
        for k in range(N):
            dt_k = dt_var[k] if dt_var is not None else self.dt
            X_k = X[:, k]
            X_next = X[:, k + 1]
            U_k = U[:, k]
            # for implicit schemes we pass U_next; handle last index by repeating
            U_next = U[:, k + 1] if k + 1 < N else U[:, k]
            self.scheme.constrain_interval(opti, time_dynamics, X_k, X_next, U_k, U_next, dt_k, extras=self.extras, k=k)

    def get_stage_duration(self, X, U, k, dt_var=None):
        if dt_var is not None:
            return dt_var[k]
        else:
            return self.dt


class SpaceSteppingTranscription(TranscriptionStrategy):
    def __init__(self, ds_array, scheme=None):
        self.ds_array = ds_array  # array of ds for each step
        from .integration_schemes import TrapezoidalScheme

        self.scheme = scheme if scheme is not None else TrapezoidalScheme()
        self.extras = None

    def initialize_variables(self, opti, horizon_len, state_size, control_size):
        X = opti.variable(state_size, horizon_len + 1)
        U = opti.variable(control_size, horizon_len)
        # create scheme-specific internal variables
        self.extras = self.scheme.create_variables(opti, horizon_len, state_size, control_size)
        return X, U, None

    def apply_dynamics(self, opti, dynamics_model, X, U, ds_array=None):
        ds = ds_array if ds_array is not None else self.ds_array

        def space_dynamics(x, u):
            # dx/ds = dx/dt * dt/ds = f(x,u) / ||v||
            v_norm = ca.norm_2(x[2:4]) + 1e-6
            dxdt = dynamics_model.state_derivative(x, u)
            return dxdt / v_norm

        N = U.shape[1]
        for k in range(N):
            ds_k = ds[k]
            X_k = X[:, k]
            X_next = X[:, k + 1]
            U_k = U[:, k]
            U_next = U[:, k + 1] if k + 1 < N else U[:, k]
            self.scheme.constrain_interval(
                opti, space_dynamics, X_k, X_next, U_k, U_next, ds_k, extras=self.extras, k=k
            )

    def get_stage_duration(self, X, U, k, ds_array=None):
        ds = ds_array if ds_array is not None else self.ds_array
        v_k = ca.norm_2(X[2:4, k])
        return ds[k] / (v_k + 1e-6)
