"""Compare CasADi dynamics and environment step outputs for a short control sequence.

Run from repository root:
    python -m tools.compare_dynamics
"""

import numpy as np
from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
from lander_learner.environment import LunarLanderEnv


def to_numpy(x):
    return np.asarray(x).reshape(-1).astype(float)


def run_compare(total_time=1.0, dt_sim=0.01, stage_dt=0.1):
    # Controls: piecewise-constant for each stage
    n_stages = int(total_time / stage_dt)
    controls = [np.array([0.5, 0.0], dtype=float) for _ in range(n_stages)]

    # initial state
    env = LunarLanderEnv(gui_enabled=False, level_name="blank")
    obs, _ = env.reset()
    # set a clear initial condition (keep altitude positive) and ensure the
    # underlying physics body matches the env state so both simulations start
    # from identical states.
    init_pos = np.array([0.0, 10.0], dtype=float)
    init_vel = np.array([0.0, 0.0], dtype=float)
    init_angle = 0.0
    init_ang_vel = 0.0
    init_fuel = float(env.initial_fuel)

    env.lander_position = init_pos.copy()
    env.lander_velocity = init_vel.copy()
    env.lander_angle = init_angle
    env.lander_angular_velocity = init_ang_vel
    env.fuel_remaining = init_fuel
    # Also update the physics body's state so the PhysicsEngine uses the same
    # starting pose when stepping.
    env.physics_engine.lander_body.position = (float(init_pos[0]), float(init_pos[1]))
    env.physics_engine.lander_body.velocity = (float(init_vel[0]), float(init_vel[1]))
    env.physics_engine.lander_body.angle = float(init_angle)
    env.physics_engine.lander_body.angular_velocity = float(init_ang_vel)

    init_state = np.hstack([env.lander_position, env.lander_velocity, env.lander_angle, env.lander_angular_velocity, env.fuel_remaining])

    # CasADi simulation
    dyn = CasadiLanderDynamics()
    cas_states = [init_state.copy()]
    cas_times = [0.0]

    t = 0.0
    for ctrl in controls:
        remaining = stage_dt
        while remaining > 1e-12:
            step = min(dt_sim, remaining)
            st = cas_states[-1]
            st_dm = dyn.discrete_step(st, ctrl, float(step))
            st_np = to_numpy(st_dm)
            cas_states.append(st_np)
            t += step
            cas_times.append(t)
            remaining -= step

    cas_states = np.column_stack(cas_states)

    # Environment simulation: reset to same initial state
    env2 = LunarLanderEnv(gui_enabled=False, level_name="blank")
    env2.reset()
    env2.lander_position = init_pos.copy()
    env2.lander_velocity = init_vel.copy()
    env2.lander_angle = init_angle
    env2.lander_angular_velocity = init_ang_vel
    env2.fuel_remaining = init_fuel
    env2.physics_engine.lander_body.position = (float(init_pos[0]), float(init_pos[1]))
    env2.physics_engine.lander_body.velocity = (float(init_vel[0]), float(init_vel[1]))
    env2.physics_engine.lander_body.angle = float(init_angle)
    env2.physics_engine.lander_body.angular_velocity = float(init_ang_vel)

    env2.time_step = dt_sim
    env_states = [np.hstack([env2.lander_position, env2.lander_velocity, env2.lander_angle, env2.lander_angular_velocity, env2.fuel_remaining])]
    env_times = [0.0]

    t = 0.0
    for ctrl in controls:
        remaining = stage_dt
        while remaining > 1e-12:
            step = min(dt_sim, remaining)
            # env.step expects action clipped in [-1,1]
            action = np.clip(ctrl, -1.0, 1.0)
            _obs, _rew, _done, _trunc, _info = env2.step(action)
            env_states.append(np.hstack([env2.lander_position, env2.lander_velocity, env2.lander_angle, env2.lander_angular_velocity, env2.fuel_remaining]))
            t += step
            env_times.append(t)
            remaining -= step

    env_states = np.column_stack(env_states)

    # Compare final states
    print('\nFinal CasADi state:\n', cas_states[:, -1])
    print('\nFinal Env state:\n', env_states[:, -1])
    diff = cas_states[:, -1] - env_states[:, -1]
    print('\nDifference (CasADi - Env):\n', diff)
    print('\nNorm of position diff:', np.linalg.norm(diff[0:2]))
    print('Norm of velocity diff:', np.linalg.norm(diff[2:4]))
    print('Angle diff:', diff[4], 'Angular vel diff:', diff[5], 'Fuel diff:', diff[6])


if __name__ == '__main__':
    run_compare()
