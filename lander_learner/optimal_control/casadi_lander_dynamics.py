"""Casadi representation of the lander free-flight dynamics.

This module provides a symbolic model of the lander motion that mirrors the
forces used by :class:`lander_learner.physics.PhysicsEngine`.  The dynamics
exclude collision handling so that optimal control solvers can impose those
constraints separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from lander_learner.utils.config import Config


@dataclass(frozen=True)
class LanderDynamicsParameters:
    """Physical parameters required to mirror the environment dynamics."""

    mass: float = Config.LANDER_MASS
    width: float = Config.LANDER_WIDTH
    height: float = Config.LANDER_HEIGHT
    gravity: float = Config.GRAVITY
    thrust_power: float = Config.THRUST_POWER
    fuel_cost: float = Config.FUEL_COST
    time_step: float = Config.TIME_STEP

    @property
    def moment_of_inertia(self) -> float:
        """Returns the planar moment of inertia for the lander body."""

        return (self.mass * (self.width**2 + self.height**2)) / 12.0


class CasadiLanderDynamics:
    """Creates Casadi functions that reproduce the environment's free-flight dynamics.

    State vector is defined as:
        [x, y, x_dot, y_dot, theta, theta_dot, fuel_remaining]"""

    STATE_SIZE = 7
    CONTROL_SIZE = 2

    def __init__(
        self,
        params: LanderDynamicsParameters | None = None,
        *,
        enforce_control_bounds: bool = True,
        enforce_fuel_limits: bool = True,
    ) -> None:
        self.params = params or LanderDynamicsParameters()
        self.enforce_control_bounds = enforce_control_bounds
        self.enforce_fuel_limits = enforce_fuel_limits

        state = ca.MX.sym("state", self.STATE_SIZE)
        control = ca.MX.sym("control", self.CONTROL_SIZE)
        dt = ca.MX.sym("dt")

        state_dot = self._state_derivative(state, control)
        self.state_derivative = ca.Function(
            "lander_state_dot",
            [state, control],
            [state_dot],
            ["state", "control"],
            ["state_dot"],
        )

        next_state = self._rk4_step(state, control, dt)
        self.discrete_step = ca.Function(
            "lander_step",
            [state, control, dt],
            [next_state],
            ["state", "control", "dt"],
            ["state_next"],
        )

    def _state_derivative(self, state: ca.MX, control: ca.MX) -> ca.MX:
        params = self.params

        velocity = state[2:4]
        angle = state[4]
        angular_velocity = state[5]
        fuel = state[6]

        if self.enforce_control_bounds:
            control = ca.fmax(ca.fmin(control, 1.0), -1.0)
        left_throttle = control[0]
        right_throttle = control[1]

        thrust_left = (left_throttle + 1.0) * 0.5 * params.thrust_power
        thrust_right = (right_throttle + 1.0) * 0.5 * params.thrust_power

        if self.enforce_fuel_limits:
            usable_fuel = ca.fmax(fuel, 0.0)
            fuel_gate = ca.if_else(usable_fuel > 0.0, 1.0, 0.0)
            thrust_left *= fuel_gate
            thrust_right *= fuel_gate

        net_thrust = thrust_left + thrust_right

        # Rotate the combined thrust into world coordinates.
        thrust_world_x = -ca.sin(angle) * net_thrust
        thrust_world_y = ca.cos(angle) * net_thrust

        acceleration = ca.vertcat(
            thrust_world_x / params.mass,
            (thrust_world_y - params.mass * params.gravity) / params.mass,
        )

        half_width = params.width * 0.5
        # Torque arises from differential thrust across the lander width.
        torque = half_width * (thrust_right - thrust_left)
        angular_acceleration = torque / params.moment_of_inertia

        fuel_rate = -(thrust_left + thrust_right) * params.fuel_cost

        state_dot = ca.vertcat(
            velocity,
            acceleration,
            angular_velocity,
            angular_acceleration,
            fuel_rate,
        )

        # Keep the position components explicit so the structure of state_dot matches state.
        return state_dot

    def _rk4_step(self, state: ca.MX, control: ca.MX, dt: ca.MX) -> ca.MX:

        k1 = self._state_derivative(state, control)
        k2 = self._state_derivative(state + 0.5 * dt * k1, control)
        k3 = self._state_derivative(state + 0.5 * dt * k2, control)
        k4 = self._state_derivative(state + dt * k3, control)

        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Ensure angular position and velocity remain scalars for downstream solvers.
        next_state = ca.vertcat(
            next_state[0:2],
            next_state[2:4],
            next_state[4],
            next_state[5],
            next_state[6],
        )

        return next_state


__all__ = ["CasadiLanderDynamics", "LanderDynamicsParameters"]
