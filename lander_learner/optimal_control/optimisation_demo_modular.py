"""Demonstration script for the modular trajectory optimisation framework.

This script builds a simple time-stepping optimiser using the provided
`TrajectoryOptimizer`, `TimeSteppingTranscription`, objectives and
constraints. It solves and saves the resulting trajectory to
`data/optimised_runs` as a NumPy archive.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lander_learner.optimal_control.transcription import TimeSteppingTranscription
from lander_learner.optimal_control.integration_schemes import TrapezoidalScheme
from lander_learner.optimal_control.trajectory_optimizer import TrajectoryOptimizer
from lander_learner.optimal_control.objectives_and_constraints import (
    MinimizeDistanceObjective,
    MinimizeControlEffortObjective,
    InitialStateConstraint,
    FinalVelocityConstraint,
    ControlBoundsConstraint,
    FuelNonnegativeConstraint,
)
from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
from lander_learner.utils.config import Config


def main() -> None:
    horizon = 200
    dt = 0.1

    dynamics = CasadiLanderDynamics()
    # Choose an integration scheme (Trapezoidal or Hermite-Simpson)
    scheme = TrapezoidalScheme()
    # scheme = HermiteSimpsonScheme()  # uncomment to try H-S collocation
    transcription = TimeSteppingTranscription(dt=dt, scheme=scheme)
    optimizer = TrajectoryOptimizer(transcription, dynamics)

    # Initial state: (x, y, vx, vy, theta, omega, fuel)
    initial_state = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, Config.INITIAL_FUEL], dtype=float)

    # Add constraints and objectives
    optimizer.add_constraint(InitialStateConstraint(initial_state))
    optimizer.add_constraint(FinalVelocityConstraint())
    optimizer.add_constraint(ControlBoundsConstraint(-1.0, 1.0))
    optimizer.add_constraint(FuelNonnegativeConstraint())

    # Use per-timestep distance objective (matches temporal demo guidance)
    optimizer.add_objective(MinimizeDistanceObjective((50.0, 10.0)))
    optimizer.add_objective(MinimizeControlEffortObjective(weight=0.01))

    # Build and solve
    optimizer.build(horizon, dynamics.STATE_SIZE, dynamics.CONTROL_SIZE)
    # Set simple initial guesses
    optimizer.opti.set_initial(optimizer.X, np.tile(initial_state.reshape(-1, 1), (1, horizon + 1)))
    optimizer.opti.set_initial(optimizer.U, 0.0)

    # If the chosen integration scheme created extra variables (e.g. H-S midpoints),
    # set sensible initial guesses for them too.
    extras = getattr(transcription, "extras", None)
    if extras:
        # common pattern: extras may contain MX variables keyed by name
        for name, var in extras.items():
            try:
                # set midpoints to the initial state repeated or zeros for controls
                if name.startswith("X"):
                    optimizer.opti.set_initial(var, np.tile(initial_state.reshape(-1, 1), (1, horizon)))
                else:
                    optimizer.opti.set_initial(var, 0.0)
            except Exception:
                # ignore if setting initial fails for any scheme-specific var
                pass

    # Print a concise problem summary before solving
    optimizer.print_problem_summary()

    # Time the solver and print elapsed time
    import time

    t0 = time.perf_counter()
    sol, X, U, dt_var = optimizer.solve()
    t_elapsed = time.perf_counter() - t0
    print(f"Solver elapsed time: {t_elapsed:.3f} s")

    states = sol.value(X)
    controls = sol.value(U)
    if dt_var is None:
        time = np.linspace(0.0, horizon * dt, horizon + 1)
    else:
        dt_values = sol.value(dt_var)
        time = np.concatenate([[0.0], np.cumsum(dt_values)])

    # Save results
    out_dir = Path(__file__).resolve().parents[2] / "data" / "optimised_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"modular_demo_{timestamp}.npz"
    np.savez(out_path, time=time, states=states, controls=controls)

    print(f"Saved modular optimisation demo to {out_path}")


if __name__ == "__main__":
    main()
