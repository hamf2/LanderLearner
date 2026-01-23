"""Modular demo using VariableTimeSteppingTranscription to run a
variable-timestep spatial optimisation equivalent to
`optimisation_demo_variable_time.py` but using the modular API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

import casadi as ca
import numpy as np

from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
from lander_learner.optimal_control.transcription import VariableTimeSteppingTranscription
from lander_learner.optimal_control.integration_schemes import TrapezoidalScheme
from lander_learner.optimal_control.trajectory_optimizer import TrajectoryOptimizer
from lander_learner.optimal_control.objectives_and_constraints import (
    ControlBoundsConstraint,
    InitialStateConstraint,
    CorridorConstraint,
    CentrelineConstraint,
    TerminalPositionConstraint,
    FuelNonnegativeConstraint,
)


def catmull_rom_chain(points: Iterable[Tuple[float, float]], samples_per_segment: int = 20) -> np.ndarray:
    pts = np.asarray(list(points), dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Need at least two control points for a spline")

    extended = np.vstack([pts[0], pts, pts[-1]])
    samples = []
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


def arc_length_parametrisation(points: np.ndarray):
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    ds = seg_lengths
    return s, ds


def resample_centreline_by_arclength(
    checkpoints: Iterable[Tuple[float, float]], num_integration_steps: int, samples_per_segment: int = 200
) -> np.ndarray:
    """Build a dense Catmull–Rom chain from `checkpoints` and resample it
    so points are evenly spaced by arc length.

    Returns an array of shape (num_integration_steps+1, 2).
    """
    dense_chain = catmull_rom_chain(checkpoints, samples_per_segment=samples_per_segment)
    s_full, _ = arc_length_parametrisation(dense_chain)
    total_len = float(s_full[-1])
    N = int(num_integration_steps) + 1
    s_target = np.linspace(0.0, total_len, N)
    cx = np.interp(s_target, s_full, dense_chain[:, 0])
    cy = np.interp(s_target, s_full, dense_chain[:, 1])
    return np.column_stack([cx, cy])


class VariableTimeSpatialConstraint:
    """Constraint enforcing positions on the centreline and discrete propagation.

    This constraint creates no new variables; it expects the optimisation's
    Opti instance to already contain the state/control/dt variables.
    """

    def __init__(self, centreline: np.ndarray, normals: np.ndarray, d_var: ca.MX):
        self.centreline = centreline
        self.normals = normals
        self.d_var = d_var

    def apply(self, opti: ca.Opti, X: ca.MX, U: ca.MX, stage_duration_fn):
        N = X.shape[1]
        K = N - 1
        for k in range(K):
            centre_k = ca.DM(self.centreline[k])
            normal_k = ca.DM(self.normals[k])
            d_k = self.d_var[k]
            opti.subject_to(X[0:2, k] == centre_k + normal_k * d_k)


def run_demo():
    # Example checkpoints (gentle S-curve)
    CHECKPOINTS = [(-20.0, 0.0), (-5.0, 20.0), (15.0, -15.0), (50.0, 0.0), (40.0, 50.0)]
    # Build a smooth centreline from checkpoints then resample it at
    # constant arc-length intervals.  Fix the number of integration steps
    # so the optimiser uses a predictable discretisation.
    INITIAL_FUEL = 120.0
    NUM_INTEGRATION_STEPS = 120

    # Resample the centreline at constant arc-length intervals.
    centreline = resample_centreline_by_arclength(CHECKPOINTS, NUM_INTEGRATION_STEPS, samples_per_segment=200)
    N = centreline.shape[0]
    print(f"Running variable-time modular demo with {NUM_INTEGRATION_STEPS} integration steps ({N} nodes)")

    dynamics = CasadiLanderDynamics()
    scheme = TrapezoidalScheme()
    trans = VariableTimeSteppingTranscription(dt=0.1, dt_min=1e-4, dt_max=10.0, scheme=scheme)

    opt = TrajectoryOptimizer(transcription=trans, dynamics_model=dynamics)

    # Create lateral offset decision variable on the optimizer's Opti
    d_var = opt.opti.variable(N)
    opt.opti.set_initial(d_var, 0.0)
    opt.add_constraint(CorridorConstraint(d_var, half_width=6.0, enforce_ends=True))

    # Attach simple constraints
    opt.add_constraint(ControlBoundsConstraint(u_min=-1.0, u_max=1.0))
    opt.add_constraint(FuelNonnegativeConstraint())
    x0 = centreline[0]
    init_state = np.array([float(x0[0]), float(x0[1]), 0.0, 0.0, 0.0, 0.0, INITIAL_FUEL])
    opt.add_constraint(InitialStateConstraint(initial_state=init_state))

    # Compute tangents and normals (needed for positional constraints)
    tangents = np.zeros((N, 2), dtype=float)
    for i in range(N):
        if i == 0:
            diff = centreline[1] - centreline[0]
        elif i == N - 1:
            diff = centreline[-1] - centreline[-2]
        else:
            diff = centreline[i + 1] - centreline[i - 1]
        norm = np.linalg.norm(diff)
        tangents[i] = diff / norm if norm >= 1e-8 else np.array([1.0, 0.0])
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Enforce positional constraints along the centreline via a single constraint
    opt.add_constraint(CentrelineConstraint(centreline, normals, d_var))

    # Terminal state constraint
    opt.add_constraint(TerminalPositionConstraint(centreline[-1]))

    # Build variables and dynamics
    opt.build(NUM_INTEGRATION_STEPS, dynamics.STATE_SIZE, dynamics.CONTROL_SIZE)

    # positional constraints are applied by CentrelineConstraint added above

    # Optional: enforce zero final velocities
    opt.opti.subject_to(opt.X[2, -1] == 0.0)
    opt.opti.subject_to(opt.X[3, -1] == 0.0)

    # Objective: minimise total time sum(dt) + small control penalty
    objective = ca.sumsqr(opt.U) * 1e-3 + ca.sum1(opt.dt_var)
    opt.opti.minimize(objective)

    # Initial guesses
    opt.opti.set_initial(opt.X, np.tile(init_state.reshape(-1, 1), (1, NUM_INTEGRATION_STEPS + 1)))
    opt.opti.set_initial(opt.U, np.zeros((dynamics.CONTROL_SIZE, NUM_INTEGRATION_STEPS)))
    opt.opti.set_initial(opt.dt_var, 0.1)

    # Solve
    opts = {"print_time": False}
    ipopt_opts = {"print_level": 0}
    sol, X_var, U_var, dt_var = opt.solve(solver_opts=opts, ipopt_opts=ipopt_opts)

    X_opt = sol.value(X_var)
    U_opt = sol.value(U_var)
    dt_opt = sol.value(dt_var)
    d_opt = sol.value(d_var)
    # Print and record final remaining fuel (state index 6)
    final_fuel = float(X_opt[6, -1])
    print(f"Final remaining fuel: {final_fuel:.6f}")

    time_nodes = np.concatenate([[0.0], np.cumsum(dt_opt)])

    out = {
        "time": time_nodes,
        "dt": dt_opt,
        "states": X_opt,
        "controls": U_opt,
        "d": d_opt,
        "centreline": centreline,
    }

    out_dir = Path(__file__).resolve().parents[2] / "data" / "optimised_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"modular_variable_time_{timestamp}.npz"
    np.savez(path, **out)
    print(f"Saved modular variable-time result to {path}")


if __name__ == "__main__":
    run_demo()
