import numpy as np
from lander_learner.observations.base_observation import BaseObservation


class DefaultObservation(BaseObservation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_size = 8  # e.g. 8 dimensions

    def get_observation(self, env):
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
