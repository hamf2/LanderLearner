"""Modular transcription strategies for lander optimal control.

This module defines the :class:`TranscriptionStrategy` abstract base
class and two concrete implementations used by the optimizer:
``TimeSteppingTranscription`` and ``SpaceSteppingTranscription``. The
strategies encapsulate how decision variables are created and how the
continuous dynamics are enforced across intervals (time or space).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Sequence, Any
import casadi as ca

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics


class TranscriptionStrategy(ABC):
    """Interface for transcription strategies.

    Subclasses must implement variable creation, dynamics enforcement and
    a stage-duration query used by modular objectives.
    """

    @abstractmethod
    def initialize_variables(
        self, opti: ca.Opti, horizon_len: int, state_size: int, control_size: int
    ) -> Tuple[ca.MX, ca.MX, Optional[ca.MX]]:
        """Create decision variables for states and controls.

        Returns a tuple ``(X, U, dt_var)``. For strategies that do not use
        a variable timestep, ``dt_var`` may be ``None``.
        """

    @abstractmethod
    def apply_dynamics(
        self, opti: ca.Opti, dynamics_model: "CasadiLanderDynamics", X: ca.MX, U: ca.MX, **kwargs: Any
    ) -> None:
        """Apply continuity/dynamics constraints to the provided ``opti``.

        The method is responsible for iterating over intervals and calling
        the chosen integration scheme to constrain each stage.
        """

    @abstractmethod
    def get_stage_duration(self, X: ca.MX, U: ca.MX, k: int, **kwargs: Any) -> ca.MX:
        """Return the (CasADi) duration associated with stage ``k``.

        This is used by objectives that need per-stage timing information.
        """


class TimeSteppingTranscription(TranscriptionStrategy):
    """Time-stepping transcription using a pluggable integration scheme.

    Args:
        dt: Default timestep used for fixed-time transcriptions.
        scheme: Integration scheme instance (provides create_variables and
            constrain_interval hooks). If ``None``, the trapezoidal scheme
            is used.
        variable_dt: If True, create a per-stage decision variable ``dt_var``.
        dt_min: Minimum allowed timestep when ``variable_dt`` is enabled.
        dt_max: Maximum allowed timestep when ``variable_dt`` is enabled.
    """

    def __init__(
        self,
        dt: float = 0.1,
        scheme: Optional[Any] = None,
        variable_dt: bool = False,
        dt_min: float = 1e-4,
        dt_max: float = 10.0,
    ) -> None:
        self.dt = dt
        self.variable_dt = variable_dt
        self.dt_min = dt_min
        self.dt_max = dt_max
        from .integration_schemes import TrapezoidalScheme

        self.scheme = scheme if scheme is not None else TrapezoidalScheme()
        self.extras: Optional[Any] = None

    def initialize_variables(
        self, opti: ca.Opti, horizon_len: int, state_size: int, control_size: int
    ) -> Tuple[ca.MX, ca.MX, Optional[ca.MX]]:
        """Create ``X`` and ``U`` decision variables and optional ``dt_var``.

        Returns:
            A tuple ``(X, U, dt_var)`` where ``dt_var`` is ``None`` for
            fixed-timestep transcriptions.
        """
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

    def apply_dynamics(
        self, opti: ca.Opti, dynamics_model: "CasadiLanderDynamics", X: ca.MX, U: ca.MX, dt_var: Optional[ca.MX] = None
    ) -> None:
        """Enforce dynamics across the horizon using the chosen scheme.

        The supplied ``dynamics_model`` is expected to provide a
        ``state_derivative(x,u)`` function compatible with CasADi.
        """

        # define continuous-time derivative wrapper
        def time_dynamics(x: ca.MX, u: ca.MX) -> ca.MX:
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

    def get_stage_duration(self, X: ca.MX, U: ca.MX, k: int, dt_var: Optional[ca.MX] = None) -> ca.MX:
        """Return the duration of stage ``k`` (CasADi MX).

        If a ``dt_var`` is provided it is used; otherwise the fixed ``dt``
        value is returned as a numeric scalar (automatically converted by
        CasADi when used in expressions).
        """
        if dt_var is not None:
            return dt_var[k]
        else:
            return self.dt


class SpaceSteppingTranscription(TranscriptionStrategy):
    """Space-stepping transcription that marches in arc length ``s``.

    Args:
        ds_array: Sequence of arc-length intervals (one per stage).
        scheme: Integration scheme to use (defaults to trapezoidal).
    """

    def __init__(self, ds_array: Sequence[float], scheme: Optional[Any] = None) -> None:
        self.ds_array = ds_array  # array of ds for each step
        from .integration_schemes import TrapezoidalScheme

        self.scheme = scheme if scheme is not None else TrapezoidalScheme()
        self.extras: Optional[Any] = None

    def initialize_variables(
        self, opti: ca.Opti, horizon_len: int, state_size: int, control_size: int
    ) -> Tuple[ca.MX, ca.MX, Optional[ca.MX]]:
        """Create state and control decision variables for space-stepping.

        The third return element is ``None`` since space-stepping uses a
        prescribed ``ds`` array rather than a variable timestep.
        """
        X = opti.variable(state_size, horizon_len + 1)
        U = opti.variable(control_size, horizon_len)
        # create scheme-specific internal variables
        self.extras = self.scheme.create_variables(opti, horizon_len, state_size, control_size)
        return X, U, None

    def apply_dynamics(
        self,
        opti: ca.Opti,
        dynamics_model: "CasadiLanderDynamics",
        X: ca.MX,
        U: ca.MX,
        ds_array: Optional[Sequence[float]] = None,
    ) -> None:
        """Enforce spatial dynamics dx/ds = f(x,u) / ||v|| across intervals.

        A small regulariser is added to the denominator to avoid division
        by zero when velocity is near zero.
        """
        ds = ds_array if ds_array is not None else self.ds_array

        def space_dynamics(x: ca.MX, u: ca.MX) -> ca.MX:
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

    def get_stage_duration(self, X: ca.MX, U: ca.MX, k: int, ds_array: Optional[Sequence[float]] = None) -> ca.MX:
        """Return approximate time duration of stage ``k`` given ``ds``.

        The duration is computed as ds / ||v|| with a small regulariser to
        avoid divide-by-zero.
        """
        ds = ds_array if ds_array is not None else self.ds_array
        v_k = ca.norm_2(X[2:4, k])
        return ds[k] / (v_k + 1e-6)
