import numpy as np
import gymnasium as gym
from gymnasium import spaces
from physics import PhysicsEngine
from utils.config import Config
from utils.rewards import get_reward_function

class LunarLanderEnv(gym.Env):
    """
    A 2D Lunar Lander Environment conforming to Gymnasium's interface.
    """

    def __init__(self, gui_enabled=False, reward_function="default"):
        super().__init__()

        self.gui_enabled = gui_enabled

        # Create a physics engine instance
        self.physics_engine = PhysicsEngine()

        # Define the observation and action space sizes.
        #   Observation: 12 variables
        #   Action: 2 thruster power values (each in range [-1, 1])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Select the reward function based on the provided name.
        self.reward_fn = get_reward_function(reward_function)

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

        self.surface_heights = np.zeros((50,), dtype=np.float32)  # Example terrain
        self.Dx = np.array(1.0, dtype=np.float32)  # distance between terrain segments
        self.target_zone = np.array([30.0, 0.0], dtype=np.float32)

        self.collision_state = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_state_variables()
        self.physics_engine.reset()

        # Return initial observation
        return self._get_observation(), {}

    def step(self, action):
        # Ensure action is within valid bounds
        action = np.clip(action, -1.0, 1.0)

        # Convert action -> thruster forces
        left_thruster, right_thruster = action

        # Update physics based on thruster forces
        self.physics_engine.update(
            left_thruster=left_thruster,
            right_thruster=right_thruster,
            env=self
        )
        self.elapsed_time += Config.TIME_STEP * Config.PHYSICS_STEPS_PER_FRAME

        # Check termination conditions (collision, out of bounds, no fuel, etc.)
        done = self._check_done()

        # Calculate reward using the selected reward function.
        reward = self._calculate_reward(done)

        # Get observation for the RL agent
        obs = self._get_observation()

        # Gymnasium step returns: obs, reward, done, truncated, info.
        # truncated=False for simplicity.
        truncated = False
        info = {"collision_state": self.collision_state}

        return obs, reward, done, truncated, info

    def _get_observation(self):
        """
        Construct and return a flattened observation vector from the current state.
        This combines:
          - lander position (2)
          - lander velocity (2)
          - angle + angular velocity (2)
          - fuel remaining (1)
          - distance to target (2)
          - altitude or other sensor readings...
        Here, we keep it simple with placeholders.
        """
        distance_to_target = self.target_zone - self.lander_position
        altitude = self.lander_position[1]
        angle = self.lander_angle % (2*np.pi)  # Keep angle in [0, 2pi)
        left_laser_distance = Config.LASER_RANGE if angle >= np.pi or angle == 0 else np.clip(altitude / np.sin(self.lander_angle), 0, Config.LASER_RANGE)
        right_laser_distance = Config.LASER_RANGE if angle <= np.pi or angle == 0 else np.clip(altitude / np.sin(-self.lander_angle), 0, Config.LASER_RANGE)

        # Flatten into a single array
        observation = np.array([
            self.lander_position[0],
            self.lander_position[1],
            self.lander_velocity[0],
            self.lander_velocity[1],
            self.lander_angle,
            self.lander_angular_velocity,
            self.fuel_remaining,
            distance_to_target[0],
            distance_to_target[1],
            left_laser_distance,
            right_laser_distance,
            float(self.collision_state),
            # Additional sensors or derived measures...
        ], dtype=np.float32)

        return observation

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
            print("Time limit reached.")
            return True

        # Collision detected
        if self.collision_state:
            print("Collision detected.")
            return True

        # If y-position is below ground baseline
        if self.lander_position[1] <= 0.0:
            self.collision_state = True
            print(f"Lander below ground. Position: x = {self.lander_position[0]:.2f}, y = {self.lander_position[1]:.2f}, "
                  f"Angle = {self.lander_angle:.2f}, Velocity: vx = {self.lander_velocity[0]:.2f}, vy = {self.lander_velocity[1]:.2f}, "
                  f"vAng = {self.lander_angular_velocity:.2f}")
            return True

        # If fuel is depleted: let it coast without fuel until it hits the ground.
        if self.fuel_remaining <= 0.0:
            return False

        return False

    def close(self):
        """
        Cleanup if needed. Called when environment is done.
        """
        pass