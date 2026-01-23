"""Visualise trajectories produced by the optimisation test script."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from lander_learner.optimal_control.casadi_lander_dynamics import CasadiLanderDynamics
from lander_learner.environment import LunarLanderEnv


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Normalises radians to the range [-pi, pi]."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def load_trajectory(file_name: str) -> dict[str, np.ndarray]:
    """Loads an optimisation result from disk, falling back to the default folder."""

    path = Path(file_name)
    if not path.suffix:
        path = path.with_suffix(".npz")

    if not path.is_file():
        base_dir = Path(__file__).resolve().parents[2] / "data" / "optimised_runs"
        path = base_dir / path.name

    if not path.is_file():
        raise FileNotFoundError(f"Could not locate optimisation file at {path}")

    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def plot_trajectory(data: dict[str, np.ndarray]) -> None:
    """Creates subplots for position, controls, heading, and speed."""

    time = data["time"]
    states = data["states"]
    controls = data["controls"]

    x = states[0]
    y = states[1]
    vx = states[2]
    vy = states[3]
    heading = _wrap_angle(states[4])
    speed = np.sqrt(vx**2 + vy**2)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    axes[0, 0].plot(x, y, label="travelled path")
    # If a centreline is present (spatial optimiser), plot it and optional checkpoints
    centreline = data.get("centreline", None)
    checkpoints = data.get("checkpoints", None)
    if centreline is not None:
        cx = centreline[:, 0]
        cy = centreline[:, 1]
        axes[0, 0].plot(cx, cy, "--", color="C2", label="centreline")
    if checkpoints is not None:
        cps = np.asarray(checkpoints)
        axes[0, 0].plot(cps[:, 0], cps[:, 1], "o", color="C3", label="checkpoints")
    axes[0, 0].set_title("Position")
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Y (m)")
    axes[0, 0].grid(True)
    pos = axes[0, 0].get_position()
    cx = pos.x0 + 0.5 * pos.width
    axes[0, 0].legend(
        loc="upper center", bbox_to_anchor=(cx, 0.95), bbox_transform=fig.transFigure, ncol=2, borderaxespad=0
    )

    axes[0, 1].plot(time[:-1], controls[0], label="Left", alpha=0.7)
    axes[0, 1].plot(time[:-1], controls[1], label="Right", alpha=0.7)
    axes[0, 1].set_title("Control Inputs")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Throttle")
    axes[0, 1].set_ylim([-1.1, 1.1])
    axes[0, 1].axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    axes[0, 1].axhline(-1.0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    pos = axes[0, 1].get_position()
    cx = pos.x0 + 0.5 * pos.width
    axes[0, 1].legend(
        loc="upper center", bbox_to_anchor=(cx, 0.95), bbox_transform=fig.transFigure, ncol=2, borderaxespad=0
    )
    axes[0, 1].grid(True)

    # Break the plot where heading wraps to avoid connecting across the ±π boundary.
    # Insert an extra NaN sample between points that wrap so the line is broken
    # cleanly between samples (rather than replacing an existing sample).
    dh = np.abs(np.diff(heading))
    wrap_idx = set(np.where(dh > np.pi)[0])
    time_plot = []
    heading_plot = []
    for i in range(len(time)):
        time_plot.append(time[i])
        heading_plot.append(heading[i])
        if i in wrap_idx:
            # insert a NaN heading and a midpoint time so the break appears
            next_t = time[i + 1] if i + 1 < len(time) else time[i]
            time_plot.append(0.5 * (time[i] + next_t))
            heading_plot.append(np.nan)

    axes[1, 0].plot(np.asarray(time_plot), np.asarray(heading_plot))
    axes[1, 0].set_title("Heading")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Angle (rad)")
    axes[1, 0].set_ylim([-np.pi, np.pi])
    axes[1, 0].grid(True)

    axes[1, 1].plot(time, speed)
    axes[1, 1].set_title("Speed")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Speed (m/s)")
    axes[1, 1].grid(True)

    # Simulate forward using the found controls to produce a "true" state
    true_states = None
    dt_arr = data.get("dt", None)
    if dt_arr is None:
        # fallback to time differences if dt not saved
        dt_arr = np.diff(time)

    try:
        # Fixed small simulation timestep for verification traces
        dt_sim = 0.01
        dyn = CasadiLanderDynamics()
        K = controls.shape[1]
        # simulate with fixed small step and record a fine trajectory
        sim_states = [np.asarray(states[:, 0]).reshape(-1)]
        sim_times = [0.0]
        t_acc = 0.0
        for k in range(K):
            ctrl = ca.DM(controls[:, k])
            remaining = float(dt_arr[k])
            while remaining > 1e-12:
                step = min(dt_sim, remaining)
                st = ca.DM(sim_states[-1])
                next_st = dyn.discrete_step(st, ctrl, float(step))
                sim_states.append(np.asarray(next_st).reshape(-1))
                t_acc += step
                sim_times.append(t_acc)
                remaining -= step

        sim_states = np.column_stack(sim_states)
        sim_times = np.asarray(sim_times)
        # trim to matching lengths (defensive - ensure arrays align)
        n_sim = min(sim_states.shape[1], sim_times.size)
        sim_states = sim_states[:, :n_sim]
        sim_times = sim_times[:n_sim]
    except Exception:
        sim_states = None
        sim_times = None

    # Overlay simulated CasADi traces on state plots using fine sim_times
    if sim_states is not None and sim_times is not None:
        axes[0, 0].plot(sim_states[0], sim_states[1], "--", color="C0", alpha=0.6, label="simulated path")
        heading_true = _wrap_angle(sim_states[4])
        # break heading plot where it wraps on the fine time grid
        dh_true = np.abs(np.diff(heading_true))
        wrap_idx_true = set(np.where(dh_true > np.pi)[0])
        time_plot_true = []
        heading_plot_true = []
        for i in range(len(sim_times)):
            time_plot_true.append(sim_times[i])
            heading_plot_true.append(heading_true[i])
            if i in wrap_idx_true:
                next_t = sim_times[i + 1] if i + 1 < len(sim_times) else sim_times[i]
                time_plot_true.append(0.5 * (sim_times[i] + next_t))
                heading_plot_true.append(np.nan)
        axes[1, 0].plot(np.asarray(time_plot_true), np.asarray(heading_plot_true), "--", color="C0", alpha=0.6)
        speed_true = np.sqrt(sim_states[2] ** 2 + sim_states[3] ** 2)
        axes[1, 1].plot(sim_times, speed_true, "--", color="C0", alpha=0.6)
        # update legends to include simulated traces (place above subplot)
        pos = axes[0, 0].get_position()
        cx = pos.x0 + 0.5 * pos.width
        axes[0, 0].legend(
            loc="upper center", bbox_to_anchor=(cx, 0.95), bbox_transform=fig.transFigure, ncol=2, borderaxespad=0
        )
        pos = axes[0, 1].get_position()
        cx = pos.x0 + 0.5 * pos.width
        axes[0, 1].legend(
            loc="upper center", bbox_to_anchor=(cx, 0.95), bbox_transform=fig.transFigure, ncol=2, borderaxespad=0
        )

    # Additionally: simulate the full environment (smaller physics timestep)
    env_states = None
    try:
        # simulate environment with same fixed small timestep
        dt_sim = 0.01
        env = LunarLanderEnv(gui_enabled=False, level_name="blank")
        env.reset()
        env.lander_position = np.asarray(states[0:2, 0]).reshape(2).astype(float)
        env.lander_velocity = np.asarray(states[2:4, 0]).reshape(2).astype(float)
        env.lander_angle = float(states[4, 0])
        env.lander_angular_velocity = float(states[5, 0])
        env.fuel_remaining = float(states[6, 0])

        # Ensure the underlying physics body starts from the same pose so the
        # environment rollout matches the CasADi discrete-step simulation from
        # the first sub-step.
        try:
            env.physics_engine.lander_body.position = (float(env.lander_position[0]), float(env.lander_position[1]))
            env.physics_engine.lander_body.velocity = (float(env.lander_velocity[0]), float(env.lander_velocity[1]))
            env.physics_engine.lander_body.angle = float(env.lander_angle)
            env.physics_engine.lander_body.angular_velocity = float(env.lander_angular_velocity)
        except Exception:
            # If the physics body isn't available for some reason, fall back
            # to the higher-level env state and continue — the try/except
            # keeps the visualiser robust.
            pass

        # temporarily override environment timestep
        original_dt = float(env.time_step)
        env.time_step = float(dt_sim)

        sim_states_env = [np.asarray(states[:, 0]).reshape(-1)]
        sim_times_env = [0.0]
        t_acc = 0.0
        K = controls.shape[1]
        for k in range(K):
            ctrl = np.asarray(controls[:, k]).reshape(-1)
            remaining = float(dt_arr[k])
            while remaining > 1e-12:
                step = min(dt_sim, remaining)
                # env.step uses env.time_step internally; ensure it matches `step`
                # we set env.time_step = dt_sim so call step once per substep
                _obs, _rew, _done, _trunc, _info = env.step(ctrl)
                t_acc += float(env.time_step)
                sim_states_env.append(
                    np.hstack(
                        [
                            env.lander_position,
                            env.lander_velocity,
                            env.lander_angle,
                            env.lander_angular_velocity,
                            env.fuel_remaining,
                        ]
                    )
                )
                sim_times_env.append(t_acc)
                remaining -= step

        # restore original timestep
        env.time_step = original_dt

        sim_states_env = np.column_stack(sim_states_env)
        sim_times_env = np.asarray(sim_times_env)
        # trim to matching lengths
        n_env = min(sim_states_env.shape[1], sim_times_env.size)
        sim_states_env = sim_states_env[:, :n_env]
        sim_times_env = sim_times_env[:n_env]
    except Exception:
        env_states = None

    # Overlay environment rollout dotted traces
    if "sim_states_env" in locals() and "sim_times_env" in locals():
        axes[0, 0].plot(sim_states_env[0], sim_states_env[1], ":", color="C0", alpha=0.5, label="env path")
        heading_env = _wrap_angle(sim_states_env[4])
        dh_env = np.abs(np.diff(heading_env))
        wrap_idx_env = set(np.where(dh_env > np.pi)[0])
        time_plot_env = []
        heading_plot_env = []
        for i in range(len(sim_times_env)):
            time_plot_env.append(sim_times_env[i])
            heading_plot_env.append(heading_env[i])
            if i in wrap_idx_env:
                next_t = sim_times_env[i + 1] if i + 1 < len(sim_times_env) else sim_times_env[i]
                time_plot_env.append(0.5 * (sim_times_env[i] + next_t))
                heading_plot_env.append(np.nan)
        axes[1, 0].plot(np.asarray(time_plot_env), np.asarray(heading_plot_env), ":", color="C0", alpha=0.5)
        speed_env = np.sqrt(sim_states_env[2] ** 2 + sim_states_env[3] ** 2)
        axes[1, 1].plot(sim_times_env, speed_env, ":", color="C0", alpha=0.5)
        pos = axes[0, 0].get_position()
        cx = pos.x0 + 0.5 * pos.width
        axes[0, 0].legend(
            loc="upper center", bbox_to_anchor=(cx, 0.95), bbox_transform=fig.transFigure, ncol=2, borderaxespad=0
        )

    # Reserve more space at the top for legends placed above each subplot
    fig.tight_layout(rect=[0, 0, 1.0, 0.9])
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise an optimised lander trajectory.")
    parser.add_argument(
        "file",
        help="Path to the optimisation output (.npz). If no directory is given,"
        " the file is assumed to reside in data/optimised_runs/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = load_trajectory(args.file)
    plot_trajectory(trajectory)


if __name__ == "__main__":
    main()
