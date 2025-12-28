"""Helpers for discovering and instantiating packaged lander levels.

The module exposes a factory that returns concrete BaseLevel implementations
based on string identifiers. Built-in options include half-plane, point-to-point,
and lap-style spline corridors, while preset lookups load JSON payloads shipped
with the package. The helper falls back to the default half-plane level when
requested names or types are missing.
"""

import logging

from .base_level import BaseLevel
from .half_plane import HalfPlaneLevel
from .lap import LapLevel, LapPresetLevel
from .point_to_point import PointToPointLevel, PointToPointPresetLevel
from .level_data import load_level_payload

logger = logging.getLogger(__name__)


def get_level(name: str = "half_plane", **kwargs) -> BaseLevel:
    """Factory returning a level instance for the requested name.

    Args:
        name (str): Canonical level identifier or preset name.
        **kwargs: Keyword arguments forwarded to the level constructor.

    Returns:
        BaseLevel: Instantiated level configured with the supplied parameters.
    """
    mapping = {
        "half_plane": HalfPlaneLevel,
        "point_to_point": PointToPointLevel,
        "point_to_point_preset": PointToPointPresetLevel,
        "lap": LapLevel,
        "lap_preset": LapPresetLevel,
    }

    key = (name or "half_plane").lower()
    level_cls = mapping.get(key)

    if level_cls is not None:
        return level_cls(**kwargs)

    try:
        payload = load_level_payload(key)
    except FileNotFoundError:
        logger.warning("Level '%s' not found; using half_plane.", name)
        return HalfPlaneLevel(**kwargs)

    level_type = payload.get("type", "point_to_point_preset").lower()
    level_cls = mapping.get(level_type)
    if level_cls is None:
        logger.warning("Level type '%s' not recognized; using half_plane.", level_type)
        level_cls = HalfPlaneLevel

    payload_kwargs = {k: v for k, v in payload.items() if k != "type"}
    payload_kwargs.update(kwargs)
    return level_cls(**payload_kwargs)


__all__ = [
    "BaseLevel",
    "HalfPlaneLevel",
    "PointToPointLevel",
    "PointToPointPresetLevel",
    "LapLevel",
    "LapPresetLevel",
    "get_level",
]
