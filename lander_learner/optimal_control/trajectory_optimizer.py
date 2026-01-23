"""Modular Trajectory Optimizer for the lander problem, supporting
pluggable transcription strategies and objectives.

This module provides :class:`TrajectoryOptimizer` which coordinates a
transcription strategy, the CasADi dynamics model and user-provided
objectives/constraints to assemble and solve a trajectory optimisation
problem using CasADi's ``Opti`` stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Tuple
import casadi as ca

if TYPE_CHECKING:
    from lander_learner.optimal_control.transcription import TranscriptionStrategy
    from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
    from lander_learner.optimal_control.objectives_and_constraints import TrajectoryObjective, TrajectoryConstraint


class TrajectoryOptimizer:
    """Coordinator for assembling and solving a trajectory optimisation.

    Args:
        transcription: A transcription strategy implementing the
            `TranscriptionStrategy` interface responsible for creating
            decision variables and applying dynamics constraints.
        dynamics_model: A CasADi-based dynamics wrapper providing
            `state_derivative` and related helper functions.
    """

    def __init__(self, transcription: "TranscriptionStrategy", dynamics_model: "CasadiLanderDynamics") -> None:
        self.transcription: "TranscriptionStrategy" = transcription
        self.dynamics: "CasadiLanderDynamics" = dynamics_model
        self.opti: ca.Opti = ca.Opti()
        self.objectives: List["TrajectoryObjective"] = []
        self.constraints: List["TrajectoryConstraint"] = []
        self.X: Optional[ca.MX] = None
        self.U: Optional[ca.MX] = None
        self.dt_var: Optional[ca.MX] = None

    def add_objective(self, obj: "TrajectoryObjective") -> None:
        """Attach an objective to the optimizer.

        The objective will be queried during :meth:`build` to obtain a
        CasADi expression which is summed into the global cost.
        """
        self.objectives.append(obj)

    def add_constraint(self, constr: "TrajectoryConstraint") -> None:
        """Attach a constraint to the optimizer.

        The constraint's :meth:`apply` method will be called during
        :meth:`build` to add constraints to the internal ``opti`` instance.
        """
        self.constraints.append(constr)

    def build(self, horizon_len: int, state_size: int, control_size: int, **kwargs) -> None:
        """Assemble the optimisation variables, dynamics and constraints.

        This calls the transcription strategy to create decision variables,
        instructs the strategy to apply dynamics constraints and then
        applies all attached objectives and constraints to the internal
        ``opti`` instance.

        Args:
            horizon_len: Number of control intervals in the horizon.
            state_size: Dimension of the state vector.
            control_size: Dimension of the control vector.
            **kwargs: Forwarded to the transcription strategy (e.g. scheme
                specific options).
        """
        # 1. Ask the Strategy to build the skeleton
        result = self.transcription.initialize_variables(self.opti, horizon_len, state_size, control_size, **kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            self.X, self.U, self.dt_var = result
        else:
            self.X, self.U = result
            self.dt_var = None

        # 2. Ask the Strategy to enforce physics
        dynamics_kwargs = dict(kwargs)
        if self.dt_var is not None:
            dynamics_kwargs["dt_var"] = self.dt_var
        self.transcription.apply_dynamics(self.opti, self.dynamics, self.X, self.U, **dynamics_kwargs)

        # 3. Apply modular Objectives/Constraints
        def stage_duration_fn(X: ca.MX, U: ca.MX, k: int) -> ca.MX:
            return self.transcription.get_stage_duration(X, U, k, dt_var=self.dt_var, **kwargs)

        for constr in self.constraints:
            constr.apply(self.opti, self.X, self.U, stage_duration_fn)

        # Collect objective expressions from objectives and minimize their sum once.
        obj_exprs: List[ca.MX] = []
        for obj in self.objectives:
            expr = obj.apply(self.opti, self.X, self.U, stage_duration_fn)
            if expr is not None:
                obj_exprs.append(expr)
        if obj_exprs:
            total_obj = obj_exprs[0]
            for e in obj_exprs[1:]:
                total_obj = total_obj + e
            self.opti.minimize(total_obj)

    def solve(
        self, solver_opts: Optional[dict] = None, ipopt_opts: Optional[dict] = None
    ) -> Tuple[object, Optional[ca.MX], Optional[ca.MX], Optional[ca.MX]]:
        """Configure the solver and solve the optimisation problem.

        Args:
            solver_opts: Options passed to ``opti.solver`` (e.g. ``{'print_time': False}``).
            ipopt_opts: IPOpt-specific options (e.g. ``{'print_level': 0}``).

        Returns:
            A tuple ``(sol, X, U, dt_var)`` where ``sol`` is the solver
            result object returned by ``opti.solve()``, and ``X``, ``U``,
            ``dt_var`` are the decision variable references created by
            :meth:`build` (may be ``None`` if not present).
        """
        if solver_opts is None:
            solver_opts = {"print_time": False}
        if ipopt_opts is None:
            ipopt_opts = {"print_level": 0}
        self.opti.solver("ipopt", solver_opts, ipopt_opts)
        sol = self.opti.solve()
        return sol, self.X, self.U, self.dt_var

    def print_problem_summary(self) -> None:
        """Print a concise summary of the optimisation problem formulation.

        Includes shapes of main decision variables, presence of variable
        timestep, and counts/names of objectives and constraints.
        """

        def _shape(var):
            if var is None:
                return None
            try:
                return (int(var.size1()), int(var.size2()))
            except Exception:
                try:
                    return tuple(var.shape)
                except Exception:
                    return str(type(var))

        X_shape = _shape(self.X)
        U_shape = _shape(self.U)
        dt_shape = _shape(self.dt_var)

        print("--- Optimisation Problem Summary ---")
        print(f"State variable `X` shape: {X_shape}")
        print(f"Control variable `U` shape: {U_shape}")
        print(f"Variable timestep present: {dt_shape is not None}")
        if dt_shape is not None:
            print(f"  dt shape: {dt_shape}")
        total_vars = 0
        try:
            if X_shape:
                total_vars += X_shape[0] * X_shape[1]
            if U_shape:
                total_vars += U_shape[0] * U_shape[1]
            if dt_shape:
                total_vars += dt_shape[0] * dt_shape[1]
        except Exception:
            total_vars = "unknown"
        print(f"Estimated decision variables: {total_vars}")
        print(f"Number of objectives attached: {len(self.objectives)}")
        if self.objectives:
            print("  Objectives:")
            for obj in self.objectives:
                print(f"    - {obj.__class__.__name__}")
        print(f"Number of constraints attached: {len(self.constraints)}")
        if self.constraints:
            print("  Constraints:")
            for c in self.constraints:
                print(f"    - {c.__class__.__name__}")
        print("-------------------------------------")
