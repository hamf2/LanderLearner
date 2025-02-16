import numpy as np
from lander_learner.observations.default_observation import DefaultObservation


class TargetObservation(DefaultObservation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_size = 13  # default (8) + 5 additional values

    def get_observation(self, env):
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
