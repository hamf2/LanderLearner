"""
Lunar Lander Environment Module

This module implements a 2D Lunar Lander environment based on Gymnasium's interface.
It simulates a lunar landing task with physics simulation, configurable rewards, custom observations,
and an optional target zone feature. The environment supports both headless and GUI modes and includes methods
for resetting the simulation, advancing one time step, and performing cleanup.
Classes:
    LunarLanderEnv: A Gymnasium-compatible environment for simulating lunar landing, managing physics, rewards,
                    observations, and termination conditions.
Usage Example:
    >>> env = LunarLanderEnv(
            gui_enabled=True, reward_function="default", observation_function="default", target_zone=True
        )
    >>> observation, info = env.reset(seed=42)
    >>> next_observation, reward, done, truncated, info = env.step([0.5, -0.2])
"""

import numpy as np
import gymnasium as gym
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple

from lander_learner.physics import PhysicsEngine
from lander_learner.utils.config import Config
from lander_learner.rewards import get_reward_class
from lander_learner.observations import get_observation_class

if TYPE_CHECKING:
    from lander_learner.observations.base_observation import BaseObservation
    from lander_learner.rewards.base_reward import BaseReward
    from lander_learner.utils.dataclasses import BodyGeometry

logger = logging.getLogger(__name__)


class LunarLanderEnv(gym.Env):
    """A 2D Lunar Lander Environment conforming to Gymnasium's interface.

    This environment simulates a 2D lunar lander using a :class:`PhysicsEngine` instance.
    It supports configurable reward and observation functions, an optional target zone,
    and termination checks that honour the active level's axis-aligned bounds.

    Attributes:
        lander_position (np.ndarray): Position of the lander centroid.
        lander_velocity (np.ndarray): Velocity of the lander centroid.
        lander_angle (np.ndarray): Orientation in radians.
        lander_angular_velocity (np.ndarray): Angular velocity in radians/s.
        fuel_remaining (np.ndarray): Remaining fuel.
        elapsed_time (np.ndarray): Episode time in seconds.
        target_position (np.ndarray): Position of the target zone centre, if enabled.
        target_zone_width (np.ndarray): Width of the target zone.
        target_zone_height (np.ndarray): Height of the target zone.
        collision_state (bool): Flag indicating if a collision occurred.
        collision_impulse (float): Impulse of the most recent collision.
        crash_state (bool): Flag indicating if a crash occurred.
        idle_state (bool): Flag indicating if the lander is idle.
        idle_timer (float): Timer for idle state.
        time_limit_reached (bool): Flag indicating if the time limit was reached.
        gui_enabled (bool): Flag indicating if GUI rendering is active.
        physics_engine (PhysicsEngine): Instance of the physics engine.
        reward (Reward): Selected reward function.
        observation (Observation): Selected observation function.
        observation_space (gym.spaces.Box): Observation space definition.
        action_space (gym.spaces.Box): Action space definition.
        level_name (str): Name of the level preset.
        target_zone (bool): Flag to enable or disable target zone.
        target_moves (bool): Flag to enable or disable target zone motion.
        finish_line (Optional[Tuple[np.ndarray, np.ndarray]]): Optional finish line segment for lap levels.
        time_step (float): Time step for each frame.
        max_episode_duration (float): Maximum duration of an episode.
        impulse_threshold (float): Threshold for collision impulse to be considered a crash.
        max_idle_time (float): Maximum idle time before termination.
        initial_fuel (float): Initial fuel amount.
    """

    def __init__(
        self,
        gui_enabled: bool = False,
        reward_function: str = "default",
        observation_function: str = "default",
        level_name: str = "half_plane",
        target_zone: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the LunarLanderEnv instance.

        Args:
            gui_enabled (bool, optional): Enables the GUI when ``True``. Defaults to ``False``.
            reward_function (str, optional): Reward function identifier. Defaults to ``"default"``.
            observation_function (str, optional): Observation function identifier. Defaults to ``"default"``.
            level_name (str, optional): Level factory key or preset. Defaults to ``"half_plane"``.
            target_zone (bool, optional): Enables the moving target zone feature. Defaults to ``False``.
            seed (int, optional): Random seed forwarded to :meth:`gym.Env.reset`.
        """
        super().__init__()

        self.gui_enabled: bool = gui_enabled

        # Create a physics engine instance.
        self.physics_engine: PhysicsEngine = PhysicsEngine(level=level_name)
        self.level_name: str = level_name

        # Select the reward function based on the provided name.
        self.reward: "BaseReward" = get_reward_class(reward_function)
        # Select the observation function and determine observation size.
        self.observation: "BaseObservation" = get_observation_class(observation_function)

        # Define the observation and action spaces.
        self.observation_space: gym.spaces.Space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation.observation_size,),
            dtype=np.float32
        )
        self.action_space: gym.spaces.Space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Set target zone mode and instantiate target zone management if enabled.
        self.target_zone: bool = target_zone
        self.target_moves: bool = False
        self.target_moves_callback: Optional[Callable[[float], np.ndarray]] = None

        # Load parameters from Config.
        self.time_step: float = Config.FRAME_TIME_STEP
        self.max_episode_duration: float = Config.MAX_EPISODE_DURATION
        self.impulse_threshold: float = Config.IMPULSE_THRESHOLD
        self.max_idle_time: float = Config.IDLE_TIMEOUT
        self.initial_fuel: float = Config.INITIAL_FUEL
        # Initialize the state by resetting the environment.
        self.reset(seed=seed)

    def reset_state_variables(self, reset_config: bool = False) -> None:
        """Resets state variables required for a new episode.

        Args:
            reset_config (bool, optional): If ``True``, reloads configuration parameters from Config.
        """
        if reset_config:
            self.time_step = Config.FRAME_TIME_STEP
            self.max_episode_duration = Config.MAX_EPISODE_DURATION
            self.impulse_threshold = Config.IMPULSE_THRESHOLD
            self.max_idle_time = Config.IDLE_TIMEOUT
            self.initial_fuel = Config.INITIAL_FUEL

        self.finish_line: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._sync_target_zone(reset_config=reset_config)
        self._sync_finish_line()
        self._sync_time_limit(reset_config=reset_config)
        self._sync_initial_fuel(reset_config=reset_config)

        self.lander_position: np.ndarray = np.array([0.0, 10.0], dtype=np.float32)
        self.lander_velocity: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)
        self.lander_angle: np.ndarray = np.array(0.0, dtype=np.float32)
        self.lander_angular_velocity: np.ndarray = np.array(0.0, dtype=np.float32)
        self.fuel_remaining: np.ndarray = np.array(self.initial_fuel, dtype=np.float32)
        self.elapsed_time: np.ndarray = np.array(0.0, dtype=np.float32)

        # Collision and state flags.
        self.collision_state: bool = False
        self.collision_impulse: float = 0.0
        self.crash_state: bool = False
        self.idle_state: bool = False
        self.idle_timer: float = 0.0
        self.lap_counter: int = 0
        self.time_limit_reached: bool = False

    def reset(self, seed: Optional[int] = None, reset_config: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state.

        Args:
            seed (int, optional): Seed for random number generation.
            reset_config (bool, optional): If ``True``, also reload configuration parameters from Config.

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): The initial observation.
                - info (dict): An empty metadata dictionary.
        """
        super().reset(seed=seed)
        self.physics_engine.reset()
        self.reset_state_variables(reset_config=reset_config)
        return self._get_observation(), {}

    def step(self, action: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs one simulation time step given an action.

        Args:
            action (Sequence[float] | np.ndarray): Thruster forces (values in [-1, 1]).

        Returns:
            tuple: A tuple (observation, reward, done, truncated, info) where:
                - observation (np.ndarray): The updated observation.
                - reward (float): The computed reward.
                - done (bool): Flag indicating whether the episode is terminated.
                - truncated (bool): Flag indicating early termination (always False here).
                - info (dict): Additional diagnostic information.
        """
        action = np.clip(action, -1.0, 1.0)
        left_thruster, right_thruster = action

        # Update physics.
        self.physics_engine.update(left_thruster=left_thruster, right_thruster=right_thruster, env=self)
        self.elapsed_time += self.time_step

        # Update target zone position if enabled.
        if self.target_moves and self.target_moves_callback is not None:
            self.target_position = self.target_moves_callback(float(self.elapsed_time))

        done = self._check_done()
        reward = self._calculate_reward(done)
        obs = self._get_observation()
        truncated = False
        info = {"collision_state": self.collision_state}

        return obs, reward, done, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Constructs and returns the current observation.

        Returns:
            np.ndarray: The observation vector.
        """
        return self.observation.get_observation(self)

    def _calculate_reward(self, done: bool) -> float:
        """Computes the reward based on the current state and termination flag.

        Args:
            done (bool): Whether the episode is terminated.

        Returns:
            float: The reward value.
        """
        return self.reward.get_reward(self, done)

    def get_level_metadata(self) -> Dict[str, Any]:
        """Exposes a copy of the active level metadata.

        Returns:
            dict: Level metadata including descriptive strings and author data.
        """
        metadata = self.physics_engine.get_level_metadata()
        return metadata

    def get_body_vertices(
        self,
        *,
        kinematic: bool = True,
        dynamic: bool = False,
        static: bool = False,
        lander: bool = False,
    ) -> "BodyGeometry":
        """Retrieves world-space geometry filtered by body type.

        Args:
            kinematic (bool, optional): Include kinematic bodies when ``True``.
            dynamic (bool, optional): Include dynamic bodies when ``True``.
            static (bool, optional): Include static bodies when ``True``.
            lander (bool, optional): Include the lander body when ``True``.

        Returns:
            BodyGeometry: Segment and polygon primitives matching the filters.
        """
        return self.physics_engine.get_body_vertices(
            kinematic=kinematic,
            dynamic=dynamic,
            static=static,
            lander=lander,
        )

    def _sync_target_zone(self, reset_config: bool) -> None:
        """Updates environment-facing target zone state from the active level.

        Args:
            reset_config (bool): If ``True``, reinitializes target zone configuration.
        """

        if self.target_zone:
            target_zone = self.physics_engine.get_target()
            if target_zone is None:
                target_zone = self.physics_engine.level.create_target_zone()
            if target_zone is not None:
                self.target_zone_moves_callback = target_zone.get_target_position
                self.target_position = target_zone.get_target_position(0.0)
                self.target_zone_width = np.array(target_zone.width, dtype=np.float32)
                self.target_zone_height = np.array(target_zone.height, dtype=np.float32)
                self.target_moves = target_zone.motion_enabled
                return

            self.target_moves_callback = lambda t: np.array([0.0, 0.0], dtype=np.float32)
        self.target_position = np.array([0.0, 0.0], dtype=np.float32)
        self.target_zone_width = np.array(0.0, dtype=np.float32)
        self.target_zone_height = np.array(0.0, dtype=np.float32)
        self.target_moves = False
        return

    def _sync_finish_line(self) -> None:
        """Caches finish line data from level metadata for GUI rendering."""

        metadata = self.physics_engine.get_level_metadata()
        finish_points = metadata.get("finish_line_points", None)

        if isinstance(finish_points, (list, tuple)) and len(finish_points) == 2:
            start = np.asarray(finish_points[0], dtype=np.float32).reshape(-1)
            end = np.asarray(finish_points[1], dtype=np.float32).reshape(-1)
            if start.size == 2 and end.size == 2:
                self.finish_line = (start.copy(), end.copy())
        else:
            self.finish_line = None

    def _sync_time_limit(self, reset_config: bool) -> None:
        """Applies per-level episode time limit overrides when available."""

        override = self.physics_engine.get_level_metadata().get("episode_time_limit", None)
        if override is not None:
            self.max_episode_duration = float(override)
            return

        if reset_config:
            self.max_episode_duration = Config.MAX_EPISODE_DURATION

    def _sync_initial_fuel(self, reset_config: bool) -> None:
        """Applies per-level initial fuel overrides when available."""

        override = self.physics_engine.get_level_metadata().get("initial_fuel", None)
        if override is not None:
            self.initial_fuel = float(override)
            return

        if reset_config:
            self.initial_fuel = Config.INITIAL_FUEL

    def _check_done(self) -> bool:
        """Determines whether the episode should terminate.

        Returns:
            bool: ``True`` if the episode should terminate.
        """
        # Check for time limit.
        if self.elapsed_time >= self.max_episode_duration:
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(
                f"Time limit reached.  Position: x = {self.lander_position[0]:.2f}, "
                f"y = {self.lander_position[1]:.2f} Angle = {angle_display:.2f} "
                f"({self.lander_angle:.2f})."
            )
            self.time_limit_reached = True
            return True

        # Check for crash due to collision impulse or unfavorable angle.
        if self.collision_state and (
            self.collision_impulse > self.impulse_threshold or
            (np.pi / 2 <= self.lander_angle % (2 * np.pi) <= 3 * np.pi / 2)
        ):
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(
                f"Crash detected.      Position: x = {self.lander_position[0]:.2f}, "
                f"y = {self.lander_position[1]:.2f}, Angle = {angle_display:.2f} "
                f"({self.lander_angle:.2f}). Impulse: {self.collision_impulse:.2f}."
            )
            self.crash_state = True
            return True

        # Check if lander is outside the level bounds.
        min_x, max_x, min_y, max_y = self.physics_engine.get_bounds()

        lander_x = float(self.lander_position[0])
        lander_y = float(self.lander_position[1])
        out_of_bounds = (
            (np.isfinite(min_x) and lander_x < min_x)
            or (np.isfinite(max_x) and lander_x > max_x)
            or (np.isfinite(min_y) and lander_y < min_y)
            or (np.isfinite(max_y) and lander_y > max_y)
        )

        if out_of_bounds:
            self.collision_state = True
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(
                f"Lander outside bounds. Position: x = {self.lander_position[0]:.2f}, "
                f"y = {self.lander_position[1]:.2f}, Angle = {angle_display:.2f} "
                f"({self.lander_angle:.2f}). Velocity: vx = {self.lander_velocity[0]:.2f}, "
                f"vy = {self.lander_velocity[1]:.2f}, vAng = {self.lander_angular_velocity:.2f}."
            )
            self.crash_state = True
            return True

        # Check for prolonged idle state.
        if self.collision_state and np.linalg.norm(self.lander_velocity) < 0.1:
            self.idle_timer += self.time_step
            if self.idle_timer > self.max_idle_time:
                angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
                logger.info(
                    f"Idle timeout. Position: x = {self.lander_position[0]:.2f}, "
                    f"y = {self.lander_position[1]:.2f}, Angle = {angle_display:.2f} "
                    f"({self.lander_angle:.2f}). Velocity: vx = {self.lander_velocity[0]:.2f}, "
                    f"vy = {self.lander_velocity[1]:.2f}, vAng = {self.lander_angular_velocity:.2f}."
                )
                self.idle_state = True
                return True
        else:
            self.idle_timer = 0.0

        if self.lap_counter >= Config.REQUIRED_LAPS:
            return True

        # If fuel is depleted, let it coast.
        if self.fuel_remaining <= 0.0:
            return False

        return False

    def close(self) -> None:
        """Performs any necessary cleanup once the environment is no longer in use."""
        pass
