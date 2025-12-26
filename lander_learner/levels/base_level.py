from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymunk


class BaseLevel(ABC):
    """Shared interface for authoring terrain layouts and scenario hooks."""

    def __init__(self, name: str, description: str = "", metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self._metadata: Dict[str, Any] = metadata.copy() if metadata else {}
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def generate_terrain(self, space: pymunk.Space):
        """Constructs static bodies (ground, obstacles) in the Pymunk space."""
        pass

    @abstractmethod
    def get_spawn_point(self) -> np.ndarray:
        """Returns the (x, y) coordinates for the lander spawn."""
        pass

    def get_spawn_rotation(self) -> float:
        """Returns the lander's starting angle in radians (default: upright)."""
        return 0.0

    def get_initial_velocity(self) -> np.ndarray:
        """Returns an optional initial (vx, vy) velocity vector."""
        return np.zeros(2, dtype=float)

    @abstractmethod
    def check_objectives(self, env) -> dict:
        """Returns specific flags (e.g., 'lap_complete', 'checkpoint_reached')."""
        pass

    def get_waypoints(self) -> List[np.ndarray]:
        """Returns ordered waypoint coordinates for routes or laps."""
        return []

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, max_x, min_y, max_y) covering the playable area."""
        inf = float("inf")
        return (-inf, inf, -inf, inf)

    def configure_environment(self, env) -> None:
        """Hook for adjusting environment parameters before the episode starts."""
        return None

    def reset(self, space: pymunk.Space) -> None:
        """Clears per-episode state; regenerates terrain when necessary."""
        self._state.clear()

    def update(self, dt: float, env=None) -> None:
        """Optional: Handle dynamic level elements (moving platforms)."""
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """Returns descriptive attributes and author-supplied metadata."""
        details = {"name": self.name, "description": self.description}
        details.update(self._metadata)
        return details.copy()
