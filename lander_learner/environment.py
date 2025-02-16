import numpy as np
import gymnasium as gym
from gymnasium import spaces
from lander_learner.physics import PhysicsEngine
from lander_learner.utils.config import Config
from lander_learner.utils.rewards import get_reward_function
from lander_learner.utils.observations import get_observation_function
from lander_learner.utils.target import TargetZone
import logging
logger = logging.getLogger(__name__)

class LunarLanderEnv(gym.Env):
    """
    A 2D Lunar Lander Environment conforming to Gymnasium's interface.
    """

    def __init__(self, gui_enabled=False, reward_function="default", observation_function="default", target_zone=False):
        super().__init__()

        self.gui_enabled = gui_enabled

        # Create a physics engine instance
        self.physics_engine = PhysicsEngine()

        # Select the reward function based on the provided name.
        self.reward_fn = get_reward_function(reward_function)
        # Select the observation function and determine observation size.
        self.observation_fn, observation_size = get_observation_function(observation_function)

        # Define the observation and action space sizes
        #   Observation: as defined by the observation function
        #   Action: 2 thruster power values (each in range [-1, 1])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Set target zone mode and instantiate target zone management if enabled.
        self.target_zone = target_zone
        self.target_moves = target_zone and Config.TARGET_ZONE_MOTION
        if self.target_zone:
            self.target_zone_obj = TargetZone()
        else:
            self.target_zone_obj = None

        # State placeholders (initially zero):
        self.reset_state_variables()

    def reset_state_variables(self):
        """
        Reset (or initialize) key state variables for a new episode.
        """
        self.lander_position = np.array([0.0, 10.0], dtype=np.float32)
        self.lander_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.lander_angle = np.array(0.0, dtype=np.float32)
        self.lander_angular_velocity = np.array(0.0, dtype=np.float32)
        self.fuel_remaining = np.array(Config.INITIAL_FUEL, dtype=np.float32)
        self.elapsed_time = np.array(0.0, dtype=np.float32)

        # Target zone parameters
        if self.target_zone:
            self.target_zone_obj.reset()
            self.target_position = self.target_zone_obj.initial_position
            self.target_zone_width = np.array(Config.TARGET_ZONE_WIDTH, dtype=np.float32)
            self.target_zone_height = np.array(Config.TARGET_ZONE_HEIGHT, dtype=np.float32)

        # Collision and state flags
        self.collision_state = False
        self.collision_impulse = 0.0
        self.crash_state = False
        self.idle_state = False
        self.idle_timer = 0.0
        self.time_limit_reached = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_state_variables()
        self.physics_engine.reset()

        # Return initial observation
        return self._get_observation(), {}

    def step(self, action):
        # Ensure action is within valid bounds
        action = np.clip(action, -1.0, 1.0)

        # Convert action to thruster forces
        left_thruster, right_thruster = action

        # Update physics based on thruster forces
        self.physics_engine.update(
            left_thruster=left_thruster,
            right_thruster=right_thruster,
            env=self
        )
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
        return self.observation_fn(self)

    def _calculate_reward(self, done):
        """
        Compute and return the reward by calling the selected reward function.
        """
        return self.reward_fn(self, done)

    def _check_done(self):
        """
        Check if the episode should terminate (e.g., collision,
        lander below ground, or other termination conditions).
        """
        # If time limit exceeded
        if self.elapsed_time >= Config.MAX_EPISODE_DURATION:
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(f"Time limit reached.  Position: x = {self.lander_position[0]:.2f}, y = {self.lander_position[1]:.2f} "
                  f"Angle = {angle_display:.2f} ({self.lander_angle:.2f}). ")
            self.time_limit_reached = True
            return True

        # Crash detected if collision impulse exceeds threshold or lander is upside down
        if self.collision_state and (
            self.collision_impulse > Config.IMPULSE_THRESHOLD or 
            (np.pi/2 <= self.lander_angle % (2*np.pi) <= 3*np.pi/2)
        ):
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(f"Crash detected.      Position: x = {self.lander_position[0]:.2f}, y = {self.lander_position[1]:.2f}, "
                  f"Angle = {angle_display:.2f} ({self.lander_angle:.2f}). Impulse: {self.collision_impulse:.2f}. ")
            self.crash_state = True
            return True

        # If y-position is below ground baseline
        if self.lander_position[1] <= 0.0:
            self.collision_state = True
            angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
            logger.info(f"Lander below ground. Position: x = {self.lander_position[0]:.2f}, y = {self.lander_position[1]:.2f}, "
                  f"Angle = {angle_display:.2f} ({self.lander_angle:.2f}). Velocity: vx = {self.lander_velocity[0]:.2f}, vy = {self.lander_velocity[1]:.2f}, "
                  f"vAng = {self.lander_angular_velocity:.2f}. ")
            self.crash_state = True
            return True
        
        # If lander is idle for too long, terminate episode
        if self.collision_state and np.linalg.norm(self.lander_velocity) < 0.1:
            self.idle_timer += Config.TIME_STEP
            if self.idle_timer > Config.IDLE_TIMEOUT:
                angle_display = ((self.lander_angle + np.pi) % (2 * np.pi)) - np.pi
                logger.info(f"Idle timeout. Position: x = {self.lander_position[0]:.2f}, y = {self.lander_position[1]:.2f}, "
                      f"Angle = {angle_display:.2f} ({self.lander_angle:.2f}). Velocity: vx = {self.lander_velocity[0]:.2f}, vy = {self.lander_velocity[1]:.2f}, "
                      f"vAng = {self.lander_angular_velocity:.2f}. ")
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