import numpy as np
from lander_learner.utils.config import Config
import logging

logger = logging.getLogger(__name__)


class TargetZone:
    """
    Encapsulates target zone placement and motion.

    The target zone can be spawned in different modes:
      - "deterministic": always at a fixed location defined in the config.
      - "on_ground": randomly placed along the ground (y = 0) within a specified x-range.
      - "above_ground": randomly placed with y > 0 within specified ranges.

    If motion is enabled (TARGET_ZONE_MOTION is True), the target zone will
    follow a piecewise linear trajectory. The trajectory is updated every
    TARGET_ZONE_MOTION_INTERVAL seconds. In each segment, a random velocity is
    sampled (within provided bounds) and used to move the target.
    """

    def __init__(self, **kwargs):
        # Read basic spawn parameters from config or kwargs
        self.spawn_mode = kwargs.get(
            "spawn_mode", Config.TARGET_ZONE_SPAWN_MODE
        )  # "deterministic", "on_ground", "above_ground"
        self.deterministic_x = kwargs.get("deterministic_x", Config.TARGET_ZONE_X)
        self.deterministic_y = kwargs.get("deterministic_y", Config.TARGET_ZONE_Y)
        self.zone_width = kwargs.get("zone_width", Config.TARGET_ZONE_WIDTH)
        self.zone_height = kwargs.get("zone_height", Config.TARGET_ZONE_HEIGHT)

        # Spawn ranges for random placement
        self.spawn_range_x = kwargs.get("spawn_range_x", Config.TARGET_ZONE_SPAWN_RANGE_X)
        self.spawn_range_y = kwargs.get("spawn_range_y", Config.TARGET_ZONE_SPAWN_RANGE_Y)

        # Motion configuration
        self.motion_enabled = kwargs.get("motion_enabled", Config.TARGET_ZONE_MOTION)
        if self.motion_enabled:
            # Use additional config parameters if available; otherwise, use defaults.
            self.motion_interval = kwargs.get("motion_interval", Config.TARGET_ZONE_MOTION_INTERVAL)
            self.vel_range_x = kwargs.get("vel_range_x", Config.TARGET_ZONE_VELOCITY_RANGE_X)
            self.vel_range_y = kwargs.get("vel_range_y", Config.TARGET_ZONE_VELOCITY_RANGE_Y)
            # Set up for piecewise linear motion.
            self.current_velocity = np.array([0.0, 0.0], dtype=np.float32)
            self.last_segment_time = np.array(0.0, dtype=np.float32)  # Time at which the current motion segment began.
        # In all cases, declare the initial target position. Run reset() to initialize random values.
        self.initial_position = np.array([0.0, 0.0], dtype=np.float32)

    def _sample_spawn_position(self):
        """
        Sample the initial target zone center position based on the spawn mode.
        """
        if self.spawn_mode == "deterministic":
            return np.array([self.deterministic_x, self.deterministic_y], dtype=np.float32)
        elif self.spawn_mode == "on_ground":
            # Place target randomly along the ground (y = 0) within the x-range.
            x = np.random.uniform(-self.spawn_range_x / 2, self.spawn_range_x / 2)
            return np.array([x, 0.0], dtype=np.float32)
        elif self.spawn_mode == "above_ground":
            # Place target randomly with y in [0, spawn_range_y] and x in the specified range.
            x = np.random.uniform(-self.spawn_range_x / 2, self.spawn_range_x / 2)
            y = np.random.uniform(0, self.spawn_range_y)
            return np.array([x, y], dtype=np.float32)
        else:
            # Fallback: use deterministic spawn.
            logger.warning(f"Unknown spawn mode: {self.spawn_mode}. Falling back to deterministic spawn.")
            return np.array([self.deterministic_x, self.deterministic_y], dtype=np.float32)

    def _sample_random_velocity(self):
        """
        Sample a random velocity vector for target motion.
        Ensure that the y-velocity is non-negative so that the target remains above ground.
        """
        vx = np.random.uniform(-self.vel_range_x, self.vel_range_x)
        vy = np.random.uniform(0, self.vel_range_y)
        return np.array([vx, vy], dtype=np.float32)

    def get_target_position(self, elapsed_time: np.ndarray) -> np.ndarray:
        """
        Given the elapsed time (in seconds), return the current target zone position.

        - If motion is disabled, the target remains at its initial position.
        - If motion is enabled, the target moves in piecewise linear segments.
          At the beginning of each segment (every `motion_interval` seconds), a new random
          velocity is sampled. The target's position is updated linearly with time within each segment.
          The position is clamped so that the target remains above the ground (y >= 0).
        """
        if not self.motion_enabled:
            return self.initial_position

        # Determine how much time has passed since the current motion segment began.
        time_in_segment = elapsed_time - self.last_segment_time
        if time_in_segment >= self.motion_interval:
            # Start a new motion segment.
            # Update the initial position to the current target position.
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
        # Ensure that the y-coordinate is not below zero.
        current_position[1] = max(current_position[1], 0.0)
        return current_position

    def reset(self):
        """
        Reset the target zone to its initial state.
        """
        self.initial_position = self._sample_spawn_position()
        if self.motion_enabled:
            self.current_velocity = self._sample_random_velocity()
            self.last_segment_time = 0.0
