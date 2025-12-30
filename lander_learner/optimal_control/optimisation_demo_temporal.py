"""Simple optimal control problem for the Lunar Lander Casadi model."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import casadi as ca
import numpy as np

from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
from lander_learner.utils.config import Config


def optimise_trajectory(
    *,
    horizon_steps: int = 200,
    dt: float = 0.1,
    start_position: tuple[float, float] = (0.0, 10.0),
    target_position: tuple[float, float] = (50.0, 10.0),
) -> dict[str, np.ndarray]:
    """Solves a minimum distance trajectory using direct multiple shooting."""

    dynamics = CasadiLanderDynamics()

    opti = ca.Opti()
    states = opti.variable(dynamics.STATE_SIZE, horizon_steps + 1)
    controls = opti.variable(dynamics.CONTROL_SIZE, horizon_steps)

    initial_state = np.array(
        [
            start_position[0],
            start_position[1],
            0.0,
            0.0,
            0.0,
            0.0,
            Config.INITIAL_FUEL,
        ],
        dtype=float,
    )
    opti.subject_to(states[:, 0] == initial_state)

    for k in range(horizon_steps):
        next_state = dynamics.discrete_step(states[:, k], controls[:, k], dt)
        opti.subject_to(states[:, k + 1] == next_state)

    opti.subject_to(opti.bounded(-1.0, controls, 1.0))
    opti.subject_to(states[6, :] >= 0.0)

    target_vec = ca.vertcat(target_position[0], target_position[1])
    target_matrix = ca.repmat(target_vec, 1, horizon_steps + 1)
    position_errors = states[0:2, :] - target_matrix
    objective = ca.sumsqr(position_errors)
    objective += 0.01 * ca.sumsqr(controls)
    opti.minimize(objective)

    opti.set_initial(states, np.tile(initial_state.reshape(-1, 1), (1, horizon_steps + 1)))
    opti.set_initial(controls, 0.0)

    solver_options = {"print_time": False}
    ipopt_options = {"print_level": 0}
    # Print concise problem summary before solving
    print("--- Optimisation Problem Summary (temporal demo) ---")
    try:
        states_shape = (int(states.size1()), int(states.size2()))
    except Exception:
        states_shape = getattr(states, "shape", None)
    try:
        controls_shape = (int(controls.size1()), int(controls.size2()))
    except Exception:
        controls_shape = getattr(controls, "shape", None)
    print(f"State variable `states` shape: {states_shape}")
    print(f"Control variable `controls` shape: {controls_shape}")
    print(f"Horizon steps: {horizon_steps}, dt: {dt}")
    print("Objective: sum squared position error across all timesteps + 0.01 * control effort")
    print("Constraints: control bounds [-1,1], fuel nonnegative on states[6,:]")
    print("-------------------------------------")

    opti.solver("ipopt", solver_options, ipopt_options)

    # Time the solver and print elapsed time
    import time

    t0 = time.perf_counter()
    solution = opti.solve()
    t_elapsed = time.perf_counter() - t0
    print(f"Solver elapsed time: {t_elapsed:.3f} s")

    solved_states = solution.value(states)
    solved_controls = solution.value(controls)
    time = np.linspace(0.0, horizon_steps * dt, horizon_steps + 1)

    return {"time": time, "states": solved_states, "controls": solved_controls}


def save_solution(data: dict[str, np.ndarray], base_path: Path) -> Path:
    """Persists optimisation results as a NumPy archive."""

    base_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    file_path = base_path / f"trajectory_{timestamp}.npz"
    np.savez(file_path, **data)
    return file_path


def main() -> None:
    result = optimise_trajectory()
    output_dir = Path(__file__).resolve().parents[2] / "data" / "optimised_runs"
    saved_path = save_solution(result, output_dir)
    print(f"Saved optimisation result to {saved_path}")


if __name__ == "__main__":
    main()
