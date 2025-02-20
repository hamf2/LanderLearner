"""
target.py

This module defines the TargetZone class, which encapsulates the placement and motion of the target zone.
The target zone can be spawned in various modes (deterministic, on_ground, or above_ground) and, if enabled,
can move in piecewise linear segments based on randomly sampled velocities.
"""

import numpy as np
from lander_learner.utils.config import Config
import logging

logger = logging.getLogger(__name__)


class TargetZone:
    """Encapsulates target zone placement and motion.

    The target zone can be spawned in different modes:
      - "deterministic": Always at a fixed location defined in the configuration.
      - "on_ground": Randomly placed along the ground (y = 0) within a specified x-range.
      - "above_ground": Randomly placed with y > 0 within specified ranges.

    If motion is enabled (TARGET_ZONE_MOTION is True), the target zone will follow a piecewise linear
    trajectory. The trajectory is updated every TARGET_ZONE_MOTION_INTERVAL seconds, during which a new
    random velocity is sampled and applied.
    """

    def __init__(self, **kwargs):
        """Initializes the TargetZone instance.

        Reads spawn parameters and motion configurations from keyword arguments or falls back to defaults
        specified in the Config. Also initializes the random number generator.

        Args:
            **kwargs: Arbitrary keyword arguments that may override configuration defaults.
                Recognized keys include:
                  - spawn_mode: The mode for spawning the target zone ("deterministic", "on_ground", "above_ground").
                  - deterministic_x: Fixed x-coordinate for deterministic spawn.
                  - deterministic_y: Fixed y-coordinate for deterministic spawn.
                  - zone_width: Width of the target zone.
                  - zone_height: Height of the target zone.
                  - spawn_range_x: Range for random x-position (for non-deterministic modes).
                  - spawn_range_y: Range for random y-position (for non-deterministic modes).
                  - motion_enabled: Flag indicating whether target motion is enabled.
                  - motion_interval: Time interval for each motion segment.
                  - vel_range_x: Range for random x-velocity during motion.
                  - vel_range_y: Range for random y-velocity during motion.
        """
        # Initialize the random generator.
        self.np_random = np.random.default_rng()

        # Read basic spawn parameters from kwargs or defaults.
        self.spawn_mode = kwargs.get("spawn_mode", Config.TARGET_ZONE_SPAWN_MODE)
        self.deterministic_x = kwargs.get("deterministic_x", Config.TARGET_ZONE_X)
        self.deterministic_y = kwargs.get("deterministic_y", Config.TARGET_ZONE_Y)
        self.zone_width = kwargs.get("zone_width", Config.TARGET_ZONE_WIDTH)
        self.zone_height = kwargs.get("zone_height", Config.TARGET_ZONE_HEIGHT)

        # Spawn ranges for random placement.
        self.spawn_range_x = kwargs.get("spawn_range_x", Config.TARGET_ZONE_SPAWN_RANGE_X)
        self.spawn_range_y = kwargs.get("spawn_range_y", Config.TARGET_ZONE_SPAWN_RANGE_Y)

        # Motion configuration.
        self.motion_enabled = kwargs.get("motion_enabled", Config.TARGET_ZONE_MOTION)
        if self.motion_enabled:
            self.motion_interval = kwargs.get("motion_interval", Config.TARGET_ZONE_MOTION_INTERVAL)
            self.vel_range_x = kwargs.get("vel_range_x", Config.TARGET_ZONE_VELOCITY_RANGE_X)
            self.vel_range_y = kwargs.get("vel_range_y", Config.TARGET_ZONE_VELOCITY_RANGE_Y)
            # Set up for piecewise linear motion.
            self.current_velocity = np.array([0.0, 0.0], dtype=np.float32)
            self.last_segment_time = np.array(0.0, dtype=np.float32)
        # Declare the initial target position; call reset() to initialize with random values.
        self.initial_position = np.array([0.0, 0.0], dtype=np.float32)

    def _sample_spawn_position(self) -> np.ndarray:
        """Samples the initial target zone center position based on the spawn mode.

        Returns:
            np.ndarray: A 2D numpy array representing the target zone's center position.
        """
        if self.spawn_mode == "deterministic":
            return np.array([self.deterministic_x, self.deterministic_y], dtype=np.float32)
        elif self.spawn_mode == "on_ground":
            # Place target randomly along the ground (y = 0) within the x-range.
            x = self.np_random.uniform(-self.spawn_range_x / 2, self.spawn_range_x / 2)
            return np.array([x, 0.0], dtype=np.float32)
        elif self.spawn_mode == "above_ground":
            # Place target randomly with y in [0, spawn_range_y] and x in the specified range.
            x = self.np_random.uniform(-self.spawn_range_x / 2, self.spawn_range_x / 2)
            y = self.np_random.uniform(0, self.spawn_range_y)
            return np.array([x, y], dtype=np.float32)
        else:
            # Fallback: use deterministic spawn.
            logger.warning(f"Unknown spawn mode: {self.spawn_mode}. Falling back to deterministic spawn.")
            return np.array([self.deterministic_x, self.deterministic_y], dtype=np.float32)

    def _sample_random_velocity(self) -> np.ndarray:
        """Samples a random velocity vector for target motion.

        Returns:
            np.ndarray: A 2D numpy array representing the random velocity vector.
        """
        vx = self.np_random.uniform(-self.vel_range_x, self.vel_range_x)
        vy = self.np_random.uniform(-self.vel_range_y if self.initial_position[1] != 0 else 0, self.vel_range_y)
        return np.array([vx, vy], dtype=np.float32)

    def get_target_position(self, elapsed_time: np.ndarray) -> np.ndarray:
        """Computes and returns the current target zone position given elapsed time.

        Args:
            elapsed_time (np.ndarray): The elapsed time (in seconds) since the start of the episode.

        Returns:
            np.ndarray: The current target zone position as a 2D numpy array.

        If motion is disabled, the target remains at its initial position.
        If motion is enabled, the target moves in piecewise linear segments.
        At the beginning of each segment (every `motion_interval` seconds), a new random velocity is sampled.
        The target's position is updated linearly within each segment and clamped so that y ≥ 0.
        """
        if not self.motion_enabled:
            return self.initial_position

        # Determine the time elapsed in the current motion segment.
        time_in_segment = elapsed_time - self.last_segment_time
        if time_in_segment >= self.motion_interval:
            # Start a new motion segment.
            self.initial_position = self.initial_position + self.current_velocity * self.motion_interval
            # Clamp the y-coordinate to ensure the target remains above the ground.
            self.initial_position[1] = max(self.initial_position[1], 0.0)
            # Reset the segment timer.
            self.last_segment_time = elapsed_time.copy()
            # Sample a new random velocity.
            self.current_velocity = self._sample_random_velocity()
            # Reset time_in_segment.
            time_in_segment = 0.0

        # Compute the current target position within the segment.
        current_position = self.initial_position + self.current_velocity * time_in_segment
        # Clamp the y-coordinate so it is not below zero.
        current_position[1] = max(current_position[1], 0.0)
        return current_position

    def reset(self, reset_config: bool = False, random_generator: np.random.Generator = None):
        """Resets the target zone to its initial state.

        Args:
            reset_config (bool, optional): If True, reinitializes configuration parameters by calling __init__().
                Defaults to False.
            random_generator (np.random.Generator, optional): An optional external random generator.
                If provided, it replaces the internal random generator. Defaults to None.

        This method re-samples the initial position based on the spawn mode and, if motion is enabled,
        resets the current velocity and segment timer.
        """
        if reset_config:
            # Reload the configuration values by reinitializing the instance.
            self.__init__()
        if random_generator is not None:
            self.np_random = random_generator
        self.initial_position = self._sample_spawn_position()
        if self.motion_enabled:
            self.current_velocity = self._sample_random_velocity()
            self.last_segment_time = 0.0
