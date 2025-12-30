"""Visualise trajectories produced by the optimisation test script."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    axes[0, 0].legend()

    axes[0, 1].plot(time[:-1], controls[0], label="Left", alpha=0.7)
    axes[0, 1].plot(time[:-1], controls[1], label="Right", alpha=0.7)
    axes[0, 1].set_title("Control Inputs")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Throttle")
    axes[0, 1].set_ylim([-1.1, 1.1])
    axes[0, 1].axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    axes[0, 1].axhline(-1.0, color="grey", linestyle="--", linewidth=1, alpha=0.8)
    axes[0, 1].legend()
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

    fig.tight_layout()
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
