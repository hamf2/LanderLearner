import logging

from .base_level import BaseLevel
from .half_plane import HalfPlaneLevel

logger = logging.getLogger(__name__)


def get_level(name: str = "half_plane", **kwargs) -> BaseLevel:
    """Factory returning a level instance for the requested name."""
    mapping = {
        "half_plane": HalfPlaneLevel,
    }

    key = (name or "half_plane").lower()
    level_cls = mapping.get(key)
    if level_cls is None:
        logger.warning("Level '%s' not found; using half_plane.", name)
        level_cls = HalfPlaneLevel

    return level_cls(**kwargs)


__all__ = ["BaseLevel", "HalfPlaneLevel", "get_level"]
