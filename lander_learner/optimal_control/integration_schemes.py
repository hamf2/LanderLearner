"""Integration schemes for transcription: Trapezoidal and Hermite-Simpson.

This module provides a small abstraction over interval integration schemes
used by the transcription layer. Implementations create any auxiliary
decision variables they need (for example midpoints for Hermite--Simpson)
and expose a single method to add the interval constraints that close the
``X_k -> X_{k+1}`` gap.

The implementations use CasADi symbolic expressions; callers should pass
an `opti` object and CASADI MX expressions for states/controls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    import casadi as ca


class IntegrationScheme(ABC):
    """Base class for integration schemes used by transcriptions.

    Subclasses should implement :meth:`create_variables` to allocate any
    per-interval decision variables and :meth:`constrain_interval` to add
    constraints that enforce the chosen integration rule on an interval.
    """

    @abstractmethod
    def create_variables(
        self, opti: ca.Opti, horizon: int, state_size: int, control_size: int
    ) -> Dict[str, ca.MX]:
        """Create auxiliary decision variables for the scheme.

        Args:
            opti: The CasADi `Opti` builder to create variables on.
            horizon: Number of intervals in the transcription.
            state_size: Dimension of the state vector.
            control_size: Dimension of the control vector.

        Returns:
            A dictionary of named CasADi MX variables (may be empty).
        """
        pass

    @abstractmethod
    def constrain_interval(
        self,
        opti: ca.Opti,
        f_dynamics: Callable[[ca.MX, ca.MX], ca.MX],
        X_k: ca.MX,
        X_next: ca.MX,
        U_k: ca.MX,
        U_next: ca.MX,
        h_step: ca.MX,
        extras: Optional[Dict[str, ca.MX]] = None,
        k: int = 0,
    ) -> None:
        """Constrain a single interval according to the integration rule.

        Args:
            opti: CasADi `Opti` instance for adding constraints.
            f_dynamics: Callable returning state derivatives given (x,u).
            X_k: State at interval start.
            X_next: State at interval end.
            U_k: Control at interval start.
            U_next: Control at interval end.
            h_step: Interval length (dt or ds) as a CasADi expression.
            extras: Optional dictionary of scheme-specific variables created
                by :meth:`create_variables`.
            k: Interval index (used by schemes which index into extras).

        The method should add constraints to ``opti`` but not return a value.
        """
        pass


class TrapezoidalScheme(IntegrationScheme):
    """Simple implicit trapezoidal integration.

    The trapezoidal rule enforces
    ``X_next = X_k + 0.5*h*(f(X_k,U_k) + f(X_next,U_next))``.
    """

    def create_variables(self, opti: ca.Opti, horizon: int, state_size: int, control_size: int) -> Dict[str, ca.MX]:
        return {}

    def constrain_interval(
        self,
        opti: ca.Opti,
        f_dynamics: Callable[[ca.MX, ca.MX], ca.MX],
        X_k: ca.MX,
        X_next: ca.MX,
        U_k: ca.MX,
        U_next: ca.MX,
        h_step: ca.MX,
        extras: Optional[Dict[str, ca.MX]] = None,
        k: int = 0,
    ) -> None:
        f_k = f_dynamics(X_k, U_k)
        f_next = f_dynamics(X_next, U_next)
        opti.subject_to(X_next == X_k + 0.5 * h_step * (f_k + f_next))


class HermiteSimpsonScheme(IntegrationScheme):
    """Hermite-Simpson collocation scheme.

    Uses a state and control midpoint for each interval and enforces both
    Simpson quadrature for integration and Hermite interpolation for the
    midpoint definition. Control midpoints are constrained to the average
    of endpoint controls to preserve a piecewise-linear control profile.
    """

    def create_variables(self, opti: "ca.Opti", horizon: int, state_size: int, control_size: int) -> Dict[str, ca.MX]:
        X_mid = opti.variable(state_size, horizon)
        U_mid = opti.variable(control_size, horizon)
        return {"X_mid": X_mid, "U_mid": U_mid}

    def constrain_interval(
        self,
        opti: "ca.Opti",
        f_dynamics: Callable[[ca.MX, ca.MX], ca.MX],
        X_k: ca.MX,
        X_next: ca.MX,
        U_k: ca.MX,
        U_next: ca.MX,
        h_step: ca.MX,
        extras: Optional[Dict[str, ca.MX]] = None,
        k: int = 0,
    ) -> None:
        if extras is None:
            raise ValueError("HermiteSimpsonScheme requires extras created by create_variables")
        X_c = extras["X_mid"][:, k]
        U_c = extras["U_mid"][:, k]

        f_k = f_dynamics(X_k, U_k)
        f_next = f_dynamics(X_next, U_next)
        f_c = f_dynamics(X_c, U_c)

        # Simpson quadrature integration constraint
        opti.subject_to(X_next == X_k + (h_step / 6.0) * (f_k + 4 * f_c + f_next))

        # Hermite interpolation constraint defining the midpoint
        opti.subject_to(X_c == 0.5 * (X_k + X_next) + (h_step / 8.0) * (f_k - f_next))

        # Enforce control midpoint as average of endpoint controls.
        opti.subject_to(U_c == 0.5 * (U_k + U_next))
