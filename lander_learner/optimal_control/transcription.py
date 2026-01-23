"""Modular transcription strategies for lander optimal control.

This module defines the :class:`TranscriptionStrategy` abstract base
class and two concrete implementations used by the optimizer:
``TimeSteppingTranscription`` and ``SpaceSteppingTranscription``. The
strategies encapsulate how decision variables are created and how the
continuous dynamics are enforced across intervals (time or space).
Modular transcription strategies for lander optimal control.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Sequence, Any, List
import casadi as ca
import numpy as np

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
        self,
        opti: ca.Opti,
        horizon_len: int,
        state_size: int,
        control_size: int,
        checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
        samples_per_segment: int = 20,
    ) -> Tuple[ca.MX, ca.MX, Optional[ca.MX]]:
        """Create decision variables for states and controls.

        Returns a tuple ``(X, U, dt_var)``. For strategies that do not use
        a variable timestep, ``dt_var`` may be ``None``.
        """

    @abstractmethod
    def apply_dynamics(
        self,
        opti: ca.Opti,
        dynamics_model: "CasadiLanderDynamics",
        X: ca.MX,
        U: ca.MX,
        checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
        samples_per_segment: int = 20,
        **kwargs: Any,
    ) -> None:
        """Apply continuity/dynamics constraints to the provided ``opti``.

        The method is responsible for iterating over intervals and calling
        the chosen integration scheme to constrain each stage.
        """

    @abstractmethod
    def get_stage_duration(
        self,
        X: ca.MX,
        U: ca.MX,
        k: int,
        checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
        samples_per_segment: int = 20,
        **kwargs: Any,
    ) -> ca.MX:
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

    def __init__(self, dt: float = 0.1, scheme: Optional[Any] = None) -> None:
        self.dt = dt
        from .integration_schemes import TrapezoidalScheme

        self.scheme = scheme if scheme is not None else TrapezoidalScheme()
        self.extras: Optional[Any] = None

    def initialize_variables(
        self,
        opti: ca.Opti,
        horizon_len: int,
        state_size: int,
        control_size: int,
        checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
        samples_per_segment: int = 20,
    ) -> Tuple[ca.MX, ca.MX, Optional[ca.MX]]:
        """Create ``X`` and ``U`` decision variables and optional ``dt_var``.

        Returns:
            A tuple ``(X, U, dt_var)`` where ``dt_var`` is ``None`` for
            fixed-timestep transcriptions.
        """
        X = opti.variable(state_size, horizon_len + 1)
        U = opti.variable(control_size, horizon_len)
        # fixed-time transcription: no per-stage dt variable
        dt_var = None

        # create scheme-specific internal variables (e.g. midpoints)
        self.extras = self.scheme.create_variables(opti, horizon_len, state_size, control_size)
        return X, U, dt_var

    def apply_dynamics(
        self,
        opti: ca.Opti,
        dynamics_model: "CasadiLanderDynamics",
        X: ca.MX,
        U: ca.MX,
        dt_var: Optional[ca.MX] = None,
        checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
        samples_per_segment: int = 20,
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


class VariableTimeSteppingTranscription(TimeSteppingTranscription):
    """Time-stepping transcription with per-stage variable `dt`.

    This class creates a per-stage `dt_var` decision variable and otherwise
    behaves like :class:`TimeSteppingTranscription`. It is intended for
    problems like `optimisation_demo_spatial.py` where `dt` is part of the
    optimisation.
    """

    def __init__(
        self, dt: float = 0.1, dt_min: float = 1e-4, dt_max: float = 10.0, scheme: Optional[Any] = None
    ) -> None:
        super().__init__(dt=dt, scheme=scheme)
        self.dt_min = dt_min
        self.dt_max = dt_max

    def initialize_variables(
        self,
        opti: ca.Opti,
        horizon_len: int,
        state_size: int,
        control_size: int,
        checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
        samples_per_segment: int = 20,
    ) -> Tuple[ca.MX, ca.MX, Optional[ca.MX]]:
        X = opti.variable(state_size, horizon_len + 1)
        U = opti.variable(control_size, horizon_len)
        dt_var = opti.variable(horizon_len)
        opti.subject_to(dt_var >= self.dt_min)
        opti.subject_to(dt_var <= self.dt_max)
        opti.set_initial(dt_var, self.dt)

        self.extras = self.scheme.create_variables(opti, horizon_len, state_size, control_size)
        return X, U, dt_var

    def apply_dynamics(
        self,
        opti: ca.Opti,
        dynamics_model: "CasadiLanderDynamics",
        X: ca.MX,
        U: ca.MX,
        dt_var: Optional[ca.MX] = None,
        **kwargs: Any,
    ) -> None:
        """Enforce discrete-time propagation using the provided per-stage `dt_var`.

        Uses the dynamics model's `discrete_step` directly to match the
        behaviour of standalone variable-time examples.
        """
        if dt_var is None:
            # fallback to parent behaviour
            return super().apply_dynamics(opti, dynamics_model, X, U, dt_var=None, **kwargs)

        N = U.shape[1]
        for k in range(N):
            X_k = X[:, k]
            X_next = X[:, k + 1]
            U_k = U[:, k]
            dt_k = dt_var[k]
            next_state = dynamics_model.discrete_step(X_k, U_k, dt_k)
            opti.subject_to(X_next == next_state)


class SpaceSteppingTranscription(TranscriptionStrategy):
    """Space-stepping transcription that marches in arc length ``s``.

    Args:
        ds_array: Sequence of arc-length intervals (one per stage).
        scheme: Integration scheme to use (defaults to trapezoidal).
    """

    def __init__(
        self, ds_array: Optional[Sequence[float]] = None, scheme: Optional[Any] = None, step_axis: int = 1
    ) -> None:
        self.ds_array = list(ds_array) if ds_array is not None else None  # array of ds for each step
        from .integration_schemes import TrapezoidalScheme

        self.scheme = scheme if scheme is not None else TrapezoidalScheme()
        self.extras: Optional[Any] = None
        self.step_axis = int(step_axis)

    def initialize_variables(
        self,
        opti: ca.Opti,
        horizon_len: int,
        state_size: int,
        control_size: int,
        checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
        samples_per_segment: int = 20,
    ) -> Tuple[ca.MX, ca.MX, Optional[ca.MX]]:
        """Create state and control decision variables for space-stepping.

        The third return element is ``None`` since space-stepping uses a
        prescribed ``ds`` array rather than a variable timestep.
        """
        X = opti.variable(state_size, horizon_len + 1)
        U = opti.variable(control_size, horizon_len)

        # create a per-stage time-decision variable to match the behaviour
        # in the standalone spatial demo which uses `dt` variables per step.
        dt_var = opti.variable(horizon_len)
        opti.subject_to(dt_var >= 1e-4)
        opti.subject_to(dt_var <= 10.0)
        opti.set_initial(dt_var, 0.1)

        # If checkpoints provided and no ds_array preset, compute centreline and ds
        if checkpoints is not None and self.ds_array is None:
            centreline = _catmull_rom_chain(checkpoints, samples_per_segment)
            _, ds = _arc_length_parametrisation(centreline)
            self.ds_array = list(ds)

        # create scheme-specific internal variables
        self.extras = self.scheme.create_variables(opti, horizon_len, state_size, control_size)
        return X, U, dt_var

    def apply_dynamics(
        self,
        opti: ca.Opti,
        dynamics_model: "CasadiLanderDynamics",
        X: ca.MX,
        U: ca.MX,
        ds_array: Optional[Sequence[float]] = None,
        checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
        samples_per_segment: int = 20,
        dt_var: Optional[ca.MX] = None,
    ) -> None:
        """Enforce spatial dynamics dx/ds = f(x,u) / ||v|| across intervals.

        A small regulariser is added to the denominator to avoid division
        by zero when velocity is near zero.
        """
        # Use provided ds_array, otherwise the transcription's ds_array
        ds = ds_array if ds_array is not None else self.ds_array
        # If checkpoints are provided and no ds_array preset, compute centreline/ds
        if checkpoints is not None and ds is None:
            centreline = _catmull_rom_chain(checkpoints, samples_per_segment)
            _, ds = _arc_length_parametrisation(centreline)
            ds = list(ds)

        # If a per-stage time variable is provided, enforce discrete-step
        # dynamics using the dynamics model's `discrete_step` function and
        # the provided `dt_var`, matching the standalone demo behaviour.
        N = U.shape[1]
        for k in range(N):
            X_k = X[:, k]
            X_next = X[:, k + 1]
            U_k = U[:, k]
            U_next = U[:, k + 1] if k + 1 < N else U[:, k]
            if dt_var is not None:
                dt_k = dt_var[k]
                next_state = dynamics_model.discrete_step(X_k, U_k, dt_k)
                opti.subject_to(X_next == next_state)
            else:
                # Fall back to spatial differential formulation if no dt_var
                def space_dynamics(x: ca.MX, u: ca.MX) -> ca.MX:
                    v_along = x[2 + self.step_axis]
                    v_scale = ca.fabs(v_along) + 1e-6
                    dxdt = dynamics_model.state_derivative(x, u)
                    return dxdt / v_scale

                ds_k = ds[k]
                self.scheme.constrain_interval(
                    opti, space_dynamics, X_k, X_next, U_k, U_next, ds_k, extras=self.extras, k=k
                )

    def get_stage_duration(
        self,
        X: ca.MX,
        U: ca.MX,
        k: int,
        ds_array: Optional[Sequence[float]] = None,
        checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
        samples_per_segment: int = 20,
        dt_var: Optional[ca.MX] = None,
    ) -> ca.MX:
        """Return approximate time duration of stage ``k`` given ``ds``.

        The duration is computed as ds / ||v|| with a small regulariser to
        avoid divide-by-zero.
        """
        ds = ds_array if ds_array is not None else self.ds_array
        # If checkpoints are provided and no ds_array preset, compute centreline/ds
        if checkpoints is not None and ds is None:
            centreline = _catmull_rom_chain(checkpoints, samples_per_segment)
            _, ds = _arc_length_parametrisation(centreline)
            ds = list(ds)

        # If a dt_var is provided by the transcription initialise step, use it.
        if dt_var is not None:
            return dt_var[k]
        v_along_k = X[2 + self.step_axis, k]
        v_scale = ca.fabs(v_along_k) + 1e-6
        return ds[k] / v_scale


def _catmull_rom_chain(points: Sequence[Tuple[float, float]], samples_per_segment: int = 20) -> np.ndarray:
    pts = np.asarray(list(points), dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Need at least two control points for a spline")

    extended = np.vstack([pts[0], pts, pts[-1]])
    samples: List[np.ndarray] = []
    for i in range(1, extended.shape[0] - 2):
        p0 = extended[i - 1]
        p1 = extended[i]
        p2 = extended[i + 1]
        p3 = extended[i + 2]
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
            b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
            c = -0.5 * p0 + 0.5 * p2
            d = p1
            point = a * t3 + b * t2 + c * t + d
            samples.append(point)
    samples.append(pts[-1])
    return np.asarray(samples)


def _arc_length_parametrisation(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    ds = seg_lengths
    return s, ds
