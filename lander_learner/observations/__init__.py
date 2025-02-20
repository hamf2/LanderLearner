"""
Observations Package

This package provides classes and functions for generating observation vectors from the Lunar Lander
environment state. It defines an abstract base class (BaseObservation) that specifies the interface
for observation generators. Concrete implementations include:
  - DefaultObservation: Generates a basic observation vector containing the lander's state.
  - TargetObservation: Extends the default observation to include information about a target zone.

Additionally, wrappers (such as NoiseObservationWrapper) are provided to modify observations (e.g., adding noise).
The factory function get_observation_class() enables easy selection of an observation generator based on a name.
"""
from .base_observation import BaseObservation
from .default_observation import DefaultObservation
from .target_observation import TargetObservation
import logging

logger = logging.getLogger(__name__)


def get_observation_class(name: str, **kwargs) -> BaseObservation:
    """Factory method to create an observation generator based on its name.

    Args:
        name (str): The name of the observation function (e.g., "default", "target").
        **kwargs: Additional keyword arguments for configuring the observation generator.

    Returns:
        BaseObservation: An instance of the requested observation generator.
    """
    mapping = {
        "default": DefaultObservation,
        "target": TargetObservation
    }
    try:
        obs_cls = mapping.get(name, DefaultObservation)
    except KeyError:
        logger.warning(f"Observation function '{name}' not found; using default observation.")
    return obs_cls(**kwargs)
