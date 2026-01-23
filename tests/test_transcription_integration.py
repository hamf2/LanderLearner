import numpy as np
import pytest

from lander_learner.optimal_control.transcription import TimeSteppingTranscription, SpaceSteppingTranscription
from lander_learner.optimal_control.integration_schemes import TrapezoidalScheme, HermiteSimpsonScheme
from lander_learner.optimal_control.trajectory_optimizer import TrajectoryOptimizer
from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
from lander_learner.optimal_control.objectives_and_constraints import (
    InitialStateConstraint,
    MinimizeDistanceObjective,
)


@pytest.mark.parametrize("scheme_cls", [TrapezoidalScheme, HermiteSimpsonScheme])
@pytest.mark.parametrize("transcription_type", ["time", "space"])
def test_transcription_scheme_combination_runs_quickly(transcription_type, scheme_cls):
    """Build and solve a tiny optimisation for each transcription/scheme combo.

    The problem is intentionally trivial: the target position equals the
    initial position so the solver should accept the initial guess and
    finish quickly. This verifies variable creation, extras handling and
    interval constraints for each combination.
    """

    horizon = 3
    state_size = CasadiLanderDynamics.STATE_SIZE
    control_size = CasadiLanderDynamics.CONTROL_SIZE

    scheme = scheme_cls()
    if transcription_type == "time":
        trans = TimeSteppingTranscription(dt=0.1, scheme=scheme)
    else:
        ds = [0.5] * horizon
        trans = SpaceSteppingTranscription(ds_array=ds, scheme=scheme)

    dyn = CasadiLanderDynamics()
    opt = TrajectoryOptimizer(trans, dyn)

    # Pick an initial state and make the target equal to the initial xy
    init_state = np.array([0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 50.0])
    opt.add_constraint(InitialStateConstraint(initial_state=init_state))
    opt.add_objective(MinimizeDistanceObjective(target_xy=(float(init_state[0]), float(init_state[1] + 50))))

    # Build the problem
    opt.build(horizon, state_size, control_size)

    # Set initial guesses for X and U (repeat initial state and zeros)
    X0 = np.tile(init_state.reshape(-1, 1), (1, horizon + 1))
    U0 = np.zeros((control_size, horizon))
    opt.opti.set_initial(opt.X, X0)
    opt.opti.set_initial(opt.U, U0)

    # If the transcription created extras (e.g. Hermite-Simpson midpoints), set sensible initials
    extras = getattr(trans, "extras", None)
    if extras:
        for name, var in extras.items():
            if name.startswith("X"):
                opt.opti.set_initial(var, X0[:, :horizon])
            else:
                # U_mid or others: set to zeros
                opt.opti.set_initial(var, np.zeros((var.size1(), var.size2())))

    # Solve with conservative IPOpt settings to keep runtime small
    solver_opts = {"print_time": False}
    ipopt_opts = {"print_level": 0, "max_iter": 200}

    sol, X_var, U_var, dt_var = opt.solve(solver_opts=solver_opts, ipopt_opts=ipopt_opts)

    # Extract solution values and verify initial node remains near the provided initial state
    X_opt = sol.value(X_var)
    assert np.allclose(X_opt[:, 0], init_state, atol=1e-6)
    # Verify the solver chose maximum thrust on both thrusters throughout
    U_opt = sol.value(U_var)
    assert np.allclose(U_opt, 1.0, atol=1e-3)
