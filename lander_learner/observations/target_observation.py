import numpy as np
from lander_learner.observations.default_observation import DefaultObservation


class TargetObservation(DefaultObservation):
    """Extended observation generator that includes target zone information.

    In addition to the default 8-dimensional observation, this generator appends 5 more
    values related to the target zone:
      - Relative x-position to the target.
      - Relative y-position to the target.
      - Angle (in radians) from the lander to the target.
      - Target zone width.
      - Target zone height.

    Attributes:
        observation_size (int): The size of the observation vector (set to 13).
    """

    def __init__(self, **kwargs):
        """Initializes the TargetObservation.

        Args:
            **kwargs: Arbitrary keyword arguments (not used in the default implementation).
        """
        super().__init__(**kwargs)
        self.observation_size = 13  # default (8 from base) + 5 additional values

    def get_observation(self, env):
        """Generates an observation vector that includes target zone information.

        The observation consists of the default observation followed by:
          - The x offset between the target and the lander.
          - The y offset between the target and the lander.
          - The angle to the target computed using arctan2.
          - The target zone's width.
          - The target zone's height.

        Args:
            env: The environment instance containing both the lander and target zone state.

        Returns:
            numpy.ndarray: A 13-dimensional observation vector.
        """
        base_obs = super().get_observation(env)
        additional = np.array(
            [
                env.target_position[0] - env.lander_position[0],
                env.target_position[1] - env.lander_position[1],
                np.arctan2(
                    env.target_position[1] - env.lander_position[1],
                    env.target_position[0] - env.lander_position[0]
                ),
                env.target_zone_width,
                env.target_zone_height,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base_obs, additional])
