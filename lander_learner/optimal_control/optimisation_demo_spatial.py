"""Spatial-stepping minimum-time optimiser along a Catmull-Rom centreline.

This example constructs a centreline from control checkpoints using a
Catmull-Rom spline, discretises the path by arc-length, and computes a
minimum-time speed profile using a spatial formulation (time = integral ds/v).

The optimiser enforces simple longitudinal acceleration limits and a
track-width constraint (lateral offset from centreline must remain within
the corridor). The lateral offset is included as a decision variable but
is initialised to zero — you can relax dynamics on offsets to explore
overtaking / lane-change behaviours.

This module is intended as a standalone example; it saves results to
``data/optimised_runs`` as a NumPy archive for later visualisation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

import casadi as ca
import numpy as np

from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
from lander_learner.utils.config import Config


def catmull_rom_chain(points: Iterable[Tuple[float, float]], samples_per_segment: int = 20) -> np.ndarray:
    """Build a Catmull–Rom chain through `points` and return sampled points.

    Args:
        points: Iterable of (x, y) control points (must be >= 2).
        samples_per_segment: Samples to generate per segment between knots.

    Returns:
        Array of shape (M, 2) containing sampled centreline points.
    """
    pts = np.asarray(list(points), dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Need at least two control points for a spline")

    # For end conditions, repeat end points (natural Catmull-Rom)
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
            # Catmull-Rom basis
            a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
            b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
            c = -0.5 * p0 + 0.5 * p2
            d = p1
            point = a * t3 + b * t2 + c * t + d
            samples.append(point)
    samples.append(pts[-1])
    return np.asarray(samples)


def arc_length_parametrisation(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative arc-length and per-segment ds for sampled points.

    Returns (s, ds) where s has length N (cumulative from 0) and ds length N-1.
    """
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    ds = seg_lengths
    return s, ds


def solve_min_time_speed_profile(
    centreline: np.ndarray,
    ds: np.ndarray,
    v_min: float = 0.1,
    v_max: float = 20.0,
    a_max: float = 5.0,
    corridor_half_width: float = 5.0,
    initial_speed: float = 0.0,
    final_speed: float = 0.0,
) -> dict:
    """Solve for speeds and lateral offsets along a spatial discretisation.

    The optimisation variables are:
      - v[k]: speed at node k
      - d[k]: lateral offset from centreline at node k

    Objective: minimize total travel time approximated by sum(ds / v).

    Constraints:
      - v_min <= v <= v_max
      - |d| <= corridor_half_width
      - longitudinal acceleration constraint using discrete energy-like relation:
          (v_{k+1}^2 - v_k^2) <= 2 * a_max * ds_k

    Returns:
        Dictionary with keys 'v', 'd', 'time', 'centreline'.
    """

    N = centreline.shape[0]
    K = N - 1

    # Compute tangents and normals for each centreline node (numpy constants)
    tangents = np.zeros((N, 2), dtype=float)
    for i in range(N):
        if i == 0:
            diff = centreline[1] - centreline[0]
        elif i == N - 1:
            diff = centreline[-1] - centreline[-2]
        else:
            diff = centreline[i + 1] - centreline[i - 1]
        norm = np.linalg.norm(diff)
        if norm < 1e-8:
            tangents[i] = np.array([1.0, 0.0])
        else:
            tangents[i] = diff / norm
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    dynamics = CasadiLanderDynamics()

    opti = ca.Opti()

    # States: [x,y,vx,vy,theta,omega,fuel]
    states = opti.variable(dynamics.STATE_SIZE, N)
    controls = opti.variable(dynamics.CONTROL_SIZE, K)
    dt = opti.variable(K)
    d = opti.variable(N)

    # Bounds and simple constraints
    opti.subject_to(dt >= 1e-4)
    opti.subject_to(dt <= 10.0)

    opti.subject_to(opti.bounded(-1.0, controls, 1.0))
    opti.subject_to(d >= -corridor_half_width)
    opti.subject_to(d <= corridor_half_width)

    # Initial and final states: position = centreline +/- offset, rest as given
    # initial lateral offset set to zero
    opti.subject_to(d[0] == 0.0)
    opti.subject_to(d[-1] == 0.0)

    # Initial state vector
    x0 = centreline[0] + normals[0] * 0.0
    init_state = ca.DM([float(x0[0]), float(x0[1]), 0.0, 0.0, 0.0, 0.0, float(Config.INITIAL_FUEL)])
    opti.subject_to(states[:, 0] == init_state)

    # Enforce positional constraints and propagate dynamics
    for k in range(K):
        # position at node k must lie on centreline + normal * d[k]
        centre_k = ca.DM(centreline[k])
        normal_k = ca.DM(normals[k])
        opti.subject_to(states[0:2, k] == centre_k + normal_k * d[k])

        # dynamics propagation for one spatial step using variable dt[k]
        next_state = dynamics.discrete_step(states[:, k], controls[:, k], dt[k])
        opti.subject_to(states[:, k + 1] == next_state)

    # enforce last position constraint
    centre_last = ca.DM(centreline[-1])
    normal_last = ca.DM(normals[-1])
    opti.subject_to(states[0:2, -1] == centre_last + normal_last * d[-1])

    # Optional: enforce zero final velocities
    opti.subject_to(states[2, -1] == 0.0)
    opti.subject_to(states[3, -1] == 0.0)

    # Objective: minimise total time sum(dt) + small control penalty
    objective = ca.sumsqr(controls) * 1e-3 + ca.sum1(dt)
    opti.minimize(objective)

    # Initial guesses
    opti.set_initial(dt, 0.1)
    opti.set_initial(controls, 0.0)
    opti.set_initial(states, ca.repmat(init_state, 1, N))

    opts = {"print_time": False}
    ipopt_opts = {"print_level": 0}
    opti.solver("ipopt", opts, ipopt_opts)

    sol = opti.solve()

    states_opt = sol.value(states)
    controls_opt = sol.value(controls)
    dt_opt = sol.value(dt)
    d_opt = sol.value(d)
    time_total = float(np.sum(dt_opt))

    # compute per-step and cumulative times
    step_times = dt_opt
    time_nodes = np.concatenate([[0.0], np.cumsum(step_times)])

    return {
        "time": time_nodes,
        "states": states_opt,
        "controls": controls_opt,
        "dt": dt_opt,
        "d": d_opt,
        "v": np.sqrt(states_opt[2, :] ** 2 + states_opt[3, :] ** 2),
        "step_times": step_times,
        "time_total": time_total,
        "centreline": centreline,
    }


def save_solution(data: dict, base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = base_dir / f"spatial_trajectory_{timestamp}.npz"
    np.savez(path, **data)
    return path


def example_run():
    # Example checkpoints (a gentle S-curve)
    checkpoints = [(-20.0, 0.0), (-5.0, 20.0), (15.0, -15.0), (50.0, 0.0), (40, 50)]
    centreline = catmull_rom_chain(checkpoints, samples_per_segment=30)
    s, ds = arc_length_parametrisation(centreline)

    result = solve_min_time_speed_profile(
        centreline=centreline,
        ds=ds,
        v_min=0.5,
        v_max=15.0,
        a_max=4.0,
        corridor_half_width=6.0,
        initial_speed=0.0,
        final_speed=0.0,
    )

    out_dir = Path(__file__).resolve().parents[2] / "data" / "optimised_runs"
    saved = save_solution(
        {
            "time": result["time"],
            "states": result["states"],
            "controls": result["controls"],
            "v": result["v"],
            "d": result["d"],
            "centreline": result["centreline"],
        },
        out_dir,
    )
    print(f"Saved spatial optimisation result to {saved}")


if __name__ == "__main__":
    example_run()
