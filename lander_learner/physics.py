"""Physics Engine module for the Lunar Lander."""

from typing import Optional

import numpy as np
import pymunk

from lander_learner.levels import BaseLevel, get_level
from lander_learner.utils.config import Config


class PhysicsEngine:
    """Manages the Chipmunk2D/pymunk-based simulation for the Lunar Lander.

    This class creates and updates the simulation space, including the lander and the ground.
    It also handles collisions by setting and resetting the collision state and recording impulses.

    Attributes:
        space (pymunk.Space): The physics simulation space.
        collision_state (bool): Indicates if a collision is currently active.
        collision_impulse (float): Records the maximum impulse from collisions.
    """

    def __init__(self, level: Optional[BaseLevel] = None):
        """Initializes the physics engine.

        Sets up the simulation space with gravity, creates the ground and the lander,
        and installs collision callbacks.
        """
        # Create the pymunk Space and set gravity.
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -Config.GRAVITY)

        self.level: BaseLevel = level or get_level()
        self.level.generate_terrain(self.space)
        self._create_lander()

        # Add collision handler callbacks.
        collision_handler = self.space.add_default_collision_handler()
        collision_handler.begin = self._collision_begin_callback
        collision_handler.separate = self._collision_separate_callback
        collision_handler.post_solve = self._collision_post_solve_callback

        self.collision_state = False
        self.collision_impulse = 0.0

    def reset(self, env=None):
        """Resets the physics state for a new episode.

        Removes the existing lander body and shape, re-creates the lander,
        refreshes the level terrain, and resets collision-related variables.
        """
        self.space.remove(self.lander_body, self.lander_shape)
        self.level.reset(self.space)
        self._create_lander()

        if env is not None:
            self.level.configure_environment(env)

        self.collision_state = False
        self.collision_impulse = 0.0

    def update(self, left_thruster, right_thruster, env):
        """Advances the physics simulation by one time step, applying thruster forces.

        Args:
            left_thruster (float): Thruster power for the left thruster (in [-1, 1]).
            right_thruster (float): Thruster power for the right thruster (in [-1, 1]).
            env: The environment instance, whose state will be updated based on the simulation.

        This method applies forces based on thruster inputs, updates fuel consumption,
        delegates level updates, steps the simulation for a fixed number of steps,
        and updates the environment's state variables.
        """
        if env.fuel_remaining > 0.0:
            # Convert thruster power to force.
            thruster_force_left = (left_thruster + 1.0) / 2.0 * Config.THRUST_POWER
            thruster_force_right = (right_thruster + 1.0) / 2.0 * Config.THRUST_POWER

            # Apply upward forces on opposite corners of the lander (in body coordinates).
            self.lander_body.apply_force_at_local_point(
                (0, thruster_force_left), (-Config.LANDER_WIDTH / 2, 0)
            )
            self.lander_body.apply_force_at_local_point(
                (0, thruster_force_right), (Config.LANDER_WIDTH / 2, 0)
            )

            # Decrease fuel in env according to consumption rate.
            fuel_used = (thruster_force_left + thruster_force_right) * Config.FUEL_COST
            env.fuel_remaining = max(0.0, env.fuel_remaining - fuel_used)

        frame_dt = Config.TIME_STEP * Config.PHYSICS_STEPS_PER_FRAME
        self.level.update(frame_dt, env=env)

        # Step the physics simulation.
        for _ in range(Config.PHYSICS_STEPS_PER_FRAME):
            self.space.step(Config.TIME_STEP)

        # Update the environment's state from the pymunk body.
        env.lander_position = np.array(self.lander_body.position, dtype=np.float32)
        env.lander_velocity = np.array(self.lander_body.velocity, dtype=np.float32)
        env.lander_angle = np.array(self.lander_body.angle, dtype=np.float32)
        env.lander_angular_velocity = np.array(self.lander_body.angular_velocity, dtype=np.float32)
        env.collision_state = self.collision_state
        env.collision_impulse = self.collision_impulse

    def _create_lander(self):
        """Creates a dynamic body for the lander.

        The lander is represented as a box with physical properties defined in Config.
        Its initial position is set, and its shape is added to the simulation space.
        """
        lander_moment = pymunk.moment_for_box(
            Config.LANDER_MASS, (Config.LANDER_WIDTH, Config.LANDER_HEIGHT)
        )
        self.lander_body = pymunk.Body(Config.LANDER_MASS, lander_moment, body_type=pymunk.Body.DYNAMIC)
        self.lander_body.sleep_threshold = 0.1
        spawn_point = self.level.get_spawn_point()
        self.lander_body.position = (float(spawn_point[0]), float(spawn_point[1]))
        self.lander_body.angle = float(self.level.get_spawn_rotation())
        self.lander_shape = pymunk.Poly.create_box(
            self.lander_body, (Config.LANDER_WIDTH, Config.LANDER_HEIGHT), radius=0.1
        )
        self.lander_shape.friction = Config.LANDER_COF
        self.lander_shape.elasticity = 0.5
        self.space.add(self.lander_body, self.lander_shape)

        initial_velocity = self.level.get_initial_velocity()
        self.lander_body.velocity = (float(initial_velocity[0]), float(initial_velocity[1]))

    def _collision_begin_callback(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict):
        """Callback invoked when collisions begin.

        Args:
            arbiter (pymunk.Arbiter): The arbiter object containing collision information.
            space (pymunk.Space): The simulation space.
            data (dict): Additional data provided by the collision handler.

        Returns:
            bool: True to process the collision.
        """
        self.collision_state = True
        return True

    def _collision_separate_callback(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict):
        """Callback invoked when collisions end.

        Args:
            arbiter (pymunk.Arbiter): The arbiter object containing collision information.
            space (pymunk.Space): The simulation space.
            data (dict): Additional data provided by the collision handler.

        Returns:
            bool: True to process the separation.
        """
        self.collision_state = False
        return True

    def _collision_post_solve_callback(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict):
        """Callback invoked after collision resolution to record the impulse.

        Args:
            arbiter (pymunk.Arbiter): The arbiter object containing collision resolution data.
            space (pymunk.Space): The simulation space.
            data (dict): Additional data provided by the collision handler.

        Returns:
            bool: True to indicate successful processing.
        """
        self.collision_impulse = max(self.collision_impulse, arbiter.total_impulse.length)
        return True
