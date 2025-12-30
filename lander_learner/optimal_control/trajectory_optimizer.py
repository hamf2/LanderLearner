"""Modular Trajectory Optimizer for the lander problem, supporting pluggable transcription strategies and objectives."""

from typing import TYPE_CHECKING, Optional, List
import casadi as ca

if TYPE_CHECKING:
    from lander_learner.optimal_control.transcription import TranscriptionStrategy
    from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
    from lander_learner.optimal_control.objectives_and_constraints import TrajectoryObjective, TrajectoryConstraint


class TrajectoryOptimizer:
    def __init__(self, transcription: "TranscriptionStrategy", dynamics_model: "CasadiLanderDynamics"):
        self.transcription: "TranscriptionStrategy" = transcription
        self.dynamics: "CasadiLanderDynamics" = dynamics_model
        self.opti: ca.Opti = ca.Opti()
        self.objectives: List["TrajectoryObjective"] = []
        self.constraints: List["TrajectoryConstraint"] = []
        self.X: Optional[ca.MX] = None
        self.U: Optional[ca.MX] = None
        self.dt_var: Optional[ca.MX] = None

    def add_objective(self, obj: "TrajectoryObjective"):
        self.objectives.append(obj)

    def add_constraint(self, constr: "TrajectoryConstraint"):
        self.constraints.append(constr)

    def build(self, horizon_len, state_size, control_size, **kwargs):
        # 1. Ask the Strategy to build the skeleton
        result = self.transcription.initialize_variables(self.opti, horizon_len, state_size, control_size)
        if isinstance(result, tuple) and len(result) == 3:
            self.X, self.U, self.dt_var = result
        else:
            self.X, self.U = result
            self.dt_var = None

        # 2. Ask the Strategy to enforce physics
        self.transcription.apply_dynamics(self.opti, self.dynamics, self.X, self.U, dt_var=self.dt_var, **kwargs)

        # 3. Apply modular Objectives/Constraints
        def stage_duration_fn(X, U, k):
            return self.transcription.get_stage_duration(X, U, k, dt_var=self.dt_var, **kwargs)

        for constr in self.constraints:
            constr.apply(self.opti, self.X, self.U, stage_duration_fn)

        # Collect objective expressions from objectives and minimize their sum once.
        obj_exprs = []
        for obj in self.objectives:
            expr = obj.apply(self.opti, self.X, self.U, stage_duration_fn)
            if expr is not None:
                obj_exprs.append(expr)
        if obj_exprs:
            total_obj = obj_exprs[0]
            for e in obj_exprs[1:]:
                total_obj = total_obj + e
            self.opti.minimize(total_obj)

    def solve(self, solver_opts=None, ipopt_opts=None):
        if solver_opts is None:
            solver_opts = {"print_time": False}
        if ipopt_opts is None:
            ipopt_opts = {"print_level": 0}
        self.opti.solver("ipopt", solver_opts, ipopt_opts)
        sol = self.opti.solve()
        return sol, self.X, self.U, self.dt_var

    def print_problem_summary(self):
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
