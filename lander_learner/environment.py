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
    """
    A 2D Lunar Lander Environment conforming to Gymnasium's interface.

    Attributes:
        gui_enabled (bool): Flag to enable or disable GUI.
        physics_engine (PhysicsEngine): Instance of the physics engine.
        reward (Reward): Selected reward function.
        observation (Observation): Selected observation function.
        observation_space (gym.spaces.Box): Observation space definition.
        action_space (gym.spaces.Box): Action space definition.
        target_zone (bool): Flag to enable or disable target zone.
        target_moves (bool): Flag to enable or disable target zone motion.
        target_zone_obj (TargetZone or None): Instance of target zone management if enabled.
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

    Methods:
        __init__(self, gui_enabled=False, reward_function="default",
        observation_function="default", target_zone=False, **kwargs):
            Initialize the Lunar Lander environment.
        reset_state_variables(self, reset_config: bool = False, target_seed: int = None):
        reset(self, seed=None, options=None):
            Reset the environment to an initial state and return the initial observation.
        step(self, action):
            Execute one time step within the environment given an action.
        _get_observation(self):
            Construct and return an observation vector from the current state using the selected observation function.
        _calculate_reward(self, done):
        _check_done(self):
            Check if the episode should terminate
            (e.g., collision, lander below ground, or other termination conditions).
        close(self):
            Cleanup if needed. Called when the environment is done.
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
        """
        Initialize the Lunar Lander environment.

        Parameters:
        gui_enabled (bool): Flag to enable or disable the graphical user interface.
        reward_function (str): Name of the reward function to use.
        observation_function (str): Name of the observation function to use.
        target_zone (bool): Flag to enable or disable the target zone feature.
        seed (int): Seed for the random number generator.
        **kwargs: Additional keyword arguments for target zone configuration.
        """
        super().__init__()

        self.gui_enabled = gui_enabled

        # Create a physics engine instance
        self.physics_engine = PhysicsEngine()

        # Select the reward function based on the provided name.
        self.reward = get_reward_class(reward_function)
        # Select the observation function and determine observation size.
        self.observation = get_observation_class(observation_function)

        # Define the observation and action space sizes
        #   Observation: as defined by the observation function
        #   Action: 2 thruster power values (each in range [-1, 1])
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

        # State placeholders (initially zero):
        self.reset(seed=seed)

    def reset_state_variables(self, reset_config=False):
        """
        Reset (or initialize) key state variables for a new episode.
        """
        if reset_config:
            self.target_moves = self.target_zone and Config.TARGET_ZONE_MOTION
        self.lander_position = np.array([0.0, 10.0], dtype=np.float32)
        self.lander_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.lander_angle = np.array(0.0, dtype=np.float32)
        self.lander_angular_velocity = np.array(0.0, dtype=np.float32)
        self.fuel_remaining = np.array(Config.INITIAL_FUEL, dtype=np.float32)
        self.elapsed_time = np.array(0.0, dtype=np.float32)

        # Target zone parameters
        if self.target_zone:
            self.target_zone_obj.reset(reset_config=reset_config, random_generator=self.np_random)
            self.target_position = self.target_zone_obj.initial_position
            self.target_zone_width = np.array(self.target_zone_obj.zone_width, dtype=np.float32)
            self.target_zone_height = np.array(self.target_zone_obj.zone_height, dtype=np.float32)

        # Collision and state flags
        self.collision_state = False
        self.collision_impulse = 0.0
        self.crash_state = False
        self.idle_state = False
        self.idle_timer = 0.0
        self.time_limit_reached = False

    def reset(self, seed=None, reset_config=False):
        super().reset(seed=seed)
        self.reset_state_variables(reset_config=reset_config)
        self.physics_engine.reset()

        # Return initial observation
        return self._get_observation(), {}

    def step(self, action):
        # Ensure action is within valid bounds
        action = np.clip(action, -1.0, 1.0)

        # Convert action to thruster forces
        left_thruster, right_thruster = action

        # Update physics based on thruster forces
        self.physics_engine.update(left_thruster=left_thruster, right_thruster=right_thruster, env=self)
        self.elapsed_time += Config.TIME_STEP * Config.PHYSICS_STEPS_PER_FRAME

        # Update target zone position if enabled
        if self.target_moves:
            self.target_position = self.target_zone_obj.get_target_position(self.elapsed_time)

        # Check termination conditions (collision, out of bounds, no fuel, etc.)
        done = self._check_done()

        # Calculate reward using the selected reward function.
        reward = self._calculate_reward(done)

        # Get observation for the RL agent.
        obs = self._get_observation()

        # Gymnasium step returns: obs, reward, done, truncated, info.
        # truncated=False for simplicity.
        truncated = False
        info = {"collision_state": self.collision_state}

        return obs, reward, done, truncated, info

    def _get_observation(self):
        """
        Construct and return an observation vector from the current state
        using the selected observation function.
        """
        return self.observation.get_observation(self)

    def _calculate_reward(self, done):
        """
        Compute and return the reward by calling the selected reward function.
        """
        return self.reward.get_reward(self, done)

    def _check_done(self):
        """
        Check if the episode should terminate (e.g., collision,
        lander below ground, or other termination conditions).
        """
        # If time limit exceeded
        if self.elapsed_time >= Config.MAX_EPISODE_DURATION:
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(
                f"Time limit reached.  Position: x = {self.lander_position[0]:.2f}, y = {self.lander_position[1]:.2f} "
                f"Angle = {angle_display:.2f} ({self.lander_angle:.2f}). "
            )
            self.time_limit_reached = True
            return True

        # Crash detected if collision impulse exceeds threshold or lander is upside down
        if self.collision_state and (
            self.collision_impulse > Config.IMPULSE_THRESHOLD
            or (np.pi / 2 <= self.lander_angle % (2 * np.pi) <= 3 * np.pi / 2)
        ):
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(
                f"Crash detected.      Position: x = {self.lander_position[0]:.2f}, y = {self.lander_position[1]:.2f}, "
                f"Angle = {angle_display:.2f} ({self.lander_angle:.2f}). Impulse: {self.collision_impulse:.2f}. "
            )
            self.crash_state = True
            return True

        # If y-position is below ground baseline
        if self.lander_position[1] <= 0.0:
            self.collision_state = True
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(
                f"Lander below ground. Position: x = {self.lander_position[0]:.2f}, y = {self.lander_position[1]:.2f}, "
                f"Angle = {angle_display:.2f} ({self.lander_angle:.2f}). Velocity: vx = {self.lander_velocity[0]:.2f}, "
                f"vy = {self.lander_velocity[1]:.2f}, vAng = {self.lander_angular_velocity:.2f}. "
            )
            self.crash_state = True
            return True

        # If lander is idle for too long, terminate episode
        if self.collision_state and np.linalg.norm(self.lander_velocity) < 0.1:
            self.idle_timer += Config.TIME_STEP
            if self.idle_timer > Config.IDLE_TIMEOUT:
                angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
                logger.info(
                    f"Idle timeout. Position: x = {self.lander_position[0]:.2f}, y = {self.lander_position[1]:.2f}, "
                    f"Angle = {angle_display:.2f} ({self.lander_angle:.2f}). "
                    f"Velocity: vx = {self.lander_velocity[0]:.2f}, vy = {self.lander_velocity[1]:.2f}, "
                    f"vAng = {self.lander_angular_velocity:.2f}. "
                )
                self.idle_state = True
                return True
        else:
            self.idle_timer = 0.0

        # If fuel is depleted: let it coast without fuel until it hits the ground.
        if self.fuel_remaining <= 0.0:
            return False

        return False

    def close(self):
        """
        Cleanup if needed. Called when environment is done.
        """
        pass
