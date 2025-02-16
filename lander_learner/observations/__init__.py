from .base_observation import BaseObservation
from .default_observation import DefaultObservation
from .target_observation import TargetObservation


def get_observation_class(name: str, **kwargs) -> BaseObservation:
    mapping = {
        "default": DefaultObservation,
        "target": TargetObservation
    }
    obs_cls = mapping.get(name, DefaultObservation)
    return obs_cls(**kwargs)
