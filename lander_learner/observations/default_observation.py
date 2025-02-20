import numpy as np
from lander_learner.observations.base_observation import BaseObservation


class DefaultObservation(BaseObservation):
    """Default observation generator for the Lunar Lander environment.

    This generator produces an 8-dimensional observation vector based on the lander's state.
    The observation includes the lander's position, velocity, orientation, angular velocity,
    fuel remaining, and collision state.

    Attributes:
        observation_size (int): The size of the observation vector (set to 8).
    """

    def __init__(self, **kwargs):
        """Initializes the DefaultObservation.

        Args:
            **kwargs: Arbitrary keyword arguments (not used in the default implementation).
        """
        super().__init__(**kwargs)
        self.observation_size = 8  # e.g., 8 dimensions

    def get_observation(self, env):
        """Generates an observation vector based on the environment state.

        The observation vector contains:
          - Lander x-position.
          - Lander y-position.
          - Lander x-velocity.
          - Lander y-velocity.
          - Lander angle (normalized between -π and π).
          - Lander angular velocity.
          - Fuel remaining.
          - Collision state as a float (0.0 or 1.0).

        Args:
            env: The environment instance containing the lander state.

        Returns:
            numpy.ndarray: An 8-dimensional observation vector of type float32.
        """
        angle = (env.lander_angle + np.pi) % (2 * np.pi) - np.pi
        observation = np.array(
            [
                env.lander_position[0],
                env.lander_position[1],
                env.lander_velocity[0],
                env.lander_velocity[1],
                angle,
                env.lander_angular_velocity,
                env.fuel_remaining,
                float(env.collision_state),
            ],
            dtype=np.float32,
        )
        return observation
