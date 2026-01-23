"""Modular spatial-stepping demo using the transcription/optimizer API.

This demo builds a centreline from checkpoints, discretises by arc-length
and uses the modular transcription API (``SpaceSteppingTranscription``)
passed a `checkpoints` argument to assemble and solve a minimum-time
speed-profile optimisation similar to `optimisation_demo_spatial.py`.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import casadi as ca

from lander_learner.optimal_control.transcription import SpaceSteppingTranscription
from lander_learner.optimal_control.integration_schemes import TrapezoidalScheme
from lander_learner.optimal_control.trajectory_optimizer import TrajectoryOptimizer
from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
from lander_learner.optimal_control.objectives_and_constraints import ControlBoundsConstraint, InitialStateConstraint
from lander_learner.utils.config import Config


def catmull_rom_chain(points: List[Tuple[float, float]], samples_per_segment: int = 20) -> np.ndarray:
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


def run_demo():
    HORIZON_LENGTH = 50
    # Example checkpoints (gentle S-curve)
    checkpoints = [(-20.0, 0.0), (-5.0, 20.0), (15.0, -15.0), (50.0, 0.0), (40.0, 50.0)]
    centreline = catmull_rom_chain(checkpoints, samples_per_segment=30)

    dynamics = CasadiLanderDynamics()
    scheme = TrapezoidalScheme()
    trans = SpaceSteppingTranscription(ds_array=None, scheme=scheme)

    opt = TrajectoryOptimizer(transcription=trans, dynamics_model=dynamics)

    # Build with horizon K and pass checkpoints so transcription can compute centreline/ds
    opt.add_constraint(ControlBoundsConstraint(u_min=-1.0, u_max=1.0))

    # set initial state and enforce it
    x0 = centreline[0]
    init_state = np.array([float(x0[0]), float(x0[1]), 0.0, 0.0, 0.0, 0.0, float(Config.INITIAL_FUEL)])
    opt.add_constraint(InitialStateConstraint(initial_state=init_state))

    # Build variables and dynamics; pass explicit ds_array so full spline is used
    opt.build(
        HORIZON_LENGTH, dynamics.STATE_SIZE, dynamics.CONTROL_SIZE, checkpoints=checkpoints, samples_per_segment=30
    )

    # Create simple total-time objective via stage durations
    total_time = 0
    if getattr(opt, "dt_var", None) is not None:
        for k in range(HORIZON_LENGTH):
            total_time = total_time + opt.dt_var[k]
    else:
        for k in range(HORIZON_LENGTH):
            total_time = total_time + trans.get_stage_duration(
                opt.X, opt.U, k, checkpoints=checkpoints, samples_per_segment=30
            )
    # small control penalty
    control_penalty = ca.sumsqr(opt.U) * 1e-3
    opt.opti.minimize(total_time + control_penalty)

    # Initial guesses
    opt.opti.set_initial(opt.X, np.tile(init_state.reshape(-1, 1), (1, HORIZON_LENGTH + 1)))
    opt.opti.set_initial(opt.U, np.zeros((dynamics.CONTROL_SIZE, HORIZON_LENGTH)))

    # Initialize scheme extras if present
    extras = getattr(trans, "extras", None)
    if extras:
        for name, var in extras.items():
            if name.startswith("X"):
                opt.opti.set_initial(var, np.tile(init_state.reshape(-1, 1), (1, HORIZON_LENGTH)))
            else:
                opt.opti.set_initial(var, np.zeros((var.size1(), var.size2())))

    # Solve
    solver_opts = {"print_time": False}
    ipopt_opts = {"print_level": 0}
    sol, X_var, U_var, dt_var = opt.solve(solver_opts=solver_opts, ipopt_opts=ipopt_opts)

    X_opt = sol.value(X_var)
    U_opt = sol.value(U_var)

    # Build time nodes from solved dt_var when present
    if dt_var is not None:
        dt_opt = sol.value(dt_var)
        time_nodes = np.concatenate([[0.0], np.cumsum(dt_opt)])
    else:
        time_nodes = np.zeros(HORIZON_LENGTH + 1)

    # Also expose `time` and `dt` keys to match the standalone demo output
    dt_opt = sol.value(dt_var) if dt_var is not None else None
    out = {
        "time": time_nodes,
        "dt": dt_opt,
        "states": X_opt,
        "controls": U_opt,
        "centreline": centreline,
        "checkpoints": np.asarray(checkpoints),
    }

    out_dir = Path(__file__).resolve().parents[2] / "data" / "optimised_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"modular_spatial_{timestamp}.npz"
    np.savez(path, **out)
    print(f"Saved modular spatial result to {path}")


if __name__ == "__main__":
    run_demo()
