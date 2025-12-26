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

from lander_learner.physics import PhysicsEngine
from lander_learner.utils.config import Config
from lander_learner.rewards import get_reward_class
from lander_learner.observations import get_observation_class
from lander_learner.utils.target import TargetZone

logger = logging.getLogger(__name__)


class LunarLanderEnv(gym.Env):
    """A 2D Lunar Lander Environment conforming to Gymnasium's interface.

    This environment simulates a 2D lunar lander using a physics engine.
    It supports configurable reward and observation functions, as well as an optional
    target zone feature whose parameters can be updated via keyword arguments.

    Attributes:
        lander_position (np.ndarray): Position of the lander.
        lander_velocity (np.ndarray): Velocity of the lander.
        lander_angle (np.ndarray): Angle of the lander.
        lander_angular_velocity (np.ndarray): Angular velocity of the lander.
        fuel_remaining (np.ndarray): Remaining fuel.
        elapsed_time (np.ndarray): Elapsed time in the episode.
        target_position (np.ndarray): Position of the target zone.
        target_zone_width (np.ndarray): Width of the target zone.
        target_zone_height (np.ndarray): Height of the target zone.
        collision_state (bool): Flag indicating if a collision occurred.
        collision_impulse (float): Impulse of the collision.
        crash_state (bool): Flag indicating if a crash occurred.
        idle_state (bool): Flag indicating if the lander is idle.
        idle_timer (float): Timer for idle state.
        time_limit_reached (bool): Flag indicating if the time limit was reached.
        gui_enabled (bool): Flag to enable or disable GUI.
        physics_engine (PhysicsEngine): Instance of the physics engine.
        reward (Reward): Selected reward function.
        observation (Observation): Selected observation function.
        observation_space (gym.spaces.Box): Observation space definition.
        action_space (gym.spaces.Box): Action space definition.
        target_zone (bool): Flag to enable or disable target zone.
        target_moves (bool): Flag to enable or disable target zone motion.
        target_zone_obj (TargetZone or None): Instance of target zone management if enabled.
        time_step (float): Time step for each frame.
        max_episode_duration (float): Maximum duration of an episode.
        impulse_threshold (float): Threshold for collision impulse to be considered a crash.
        max_idle_time (float): Maximum idle time before termination.
        initial_fuel (float): Initial fuel amount.
    """

    def __init__(
        self,
        gui_enabled=False,
        reward_function="default",
        observation_function="default",
        target_zone=False,
        seed=None,
        **kwargs
    ):
        """Initializes the LunarLanderEnv instance.

        Args:
            gui_enabled (bool, optional): Enables or disables the graphical user interface.
            reward_function (str, optional): Specifies the reward function to be employed.
            observation_function (str, optional): Specifies the observation function used to
                construct the observation vector.
            target_zone (bool, optional): Enables or disables the target zone feature.
            seed (int, optional): Seed for random number generation.
            **kwargs: Additional keyword arguments passed to the TargetZone constructor.
        """
        super().__init__()

        self.gui_enabled = gui_enabled

        # Create a physics engine instance.
        self.physics_engine = PhysicsEngine()

        # Select the reward function based on the provided name.
        self.reward = get_reward_class(reward_function)
        # Select the observation function and determine observation size.
        self.observation = get_observation_class(observation_function)

        # Define the observation and action spaces.
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation.observation_size,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Set target zone mode and instantiate target zone management if enabled.
        self.target_zone = target_zone
        self.target_moves = target_zone and Config.TARGET_ZONE_MOTION
        if self.target_zone:
            self.target_zone_obj = TargetZone(**kwargs)
        else:
            self.target_zone_obj = None

        # Load parameters from Config.
        self.time_step = Config.FRAME_TIME_STEP
        self.max_episode_duration = Config.MAX_EPISODE_DURATION
        self.impulse_threshold = Config.IMPULSE_THRESHOLD
        self.max_idle_time = Config.IDLE_TIMEOUT
        self.initial_fuel = Config.INITIAL_FUEL

        # Initialize the state by resetting the environment.
        self.reset(seed=seed)

    def reset_state_variables(self, reset_config: bool = False):
        """Resets state variables required for a new episode.

        Args:
            reset_config (bool, optional): If True, reloads configuration parameters from Config.
        """
        if reset_config:
            self.target_moves = self.target_zone and Config.TARGET_ZONE_MOTION
            self.time_step = Config.FRAME_TIME_STEP
            self.max_episode_duration = Config.MAX_EPISODE_DURATION
            self.impulse_threshold = Config.IMPULSE_THRESHOLD
            self.max_idle_time = Config.IDLE_TIMEOUT
            self.initial_fuel = Config.INITIAL_FUEL

        self.lander_position = np.array([0.0, 10.0], dtype=np.float32)
        self.lander_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.lander_angle = np.array(0.0, dtype=np.float32)
        self.lander_angular_velocity = np.array(0.0, dtype=np.float32)
        self.fuel_remaining = np.array(self.initial_fuel, dtype=np.float32)
        self.elapsed_time = np.array(0.0, dtype=np.float32)

        # Target zone parameters.
        if self.target_zone:
            self.target_zone_obj.reset(reset_config=reset_config, random_generator=self.np_random)
            self.target_position = self.target_zone_obj.initial_position
            self.target_zone_width = np.array(self.target_zone_obj.zone_width, dtype=np.float32)
            self.target_zone_height = np.array(self.target_zone_obj.zone_height, dtype=np.float32)

        # Collision and state flags.
        self.collision_state = False
        self.collision_impulse = 0.0
        self.crash_state = False
        self.idle_state = False
        self.idle_timer = 0.0
        self.time_limit_reached = False

    def reset(self, seed=None, reset_config=False):
        """Resets the environment to an initial state.

        Args:
            seed (int, optional): Seed for random number generation.
            reset_config (bool, optional): If True, also reload configuration parameters from Config.

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): The initial observation.
                - info (dict): An empty metadata dictionary.
        """
        super().reset(seed=seed)
        self.reset_state_variables(reset_config=reset_config)
        self.physics_engine.reset(env=self)
        return self._get_observation(), {}

    def step(self, action):
        """Performs one simulation time step given an action.

        Args:
            action (np.ndarray): Action representing thruster forces (values in [-1, 1]).

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
        if self.target_moves:
            self.target_position = self.target_zone_obj.get_target_position(self.elapsed_time)

        done = self._check_done()
        reward = self._calculate_reward(done)
        obs = self._get_observation()
        truncated = False
        info = {"collision_state": self.collision_state}

        return obs, reward, done, truncated, info

    def _get_observation(self):
        """Constructs and returns the current observation.

        Returns:
            np.ndarray: The observation vector.
        """
        return self.observation.get_observation(self)

    def _calculate_reward(self, done):
        """Computes the reward based on the current state and termination flag.

        Args:
            done (bool): Whether the episode is terminated.

        Returns:
            float: The reward value.
        """
        return self.reward.get_reward(self, done)

    def _check_done(self):
        """Determines whether the episode should terminate.

        Returns:
            bool: True if termination conditions are met, otherwise False.
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

        # Check if lander is below ground.
        if self.lander_position[1] <= 0.0:
            self.collision_state = True
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(
                f"Lander below ground. Position: x = {self.lander_position[0]:.2f}, "
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

        # If fuel is depleted, let it coast.
        if self.fuel_remaining <= 0.0:
            return False

        return False

    def close(self):
        """Performs any necessary cleanup once the environment is no longer in use."""
        pass
