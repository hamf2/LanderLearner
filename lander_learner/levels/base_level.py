from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymunk


class BaseLevel(ABC):
    """Shared interface for authoring terrain layouts and scenario hooks.

    Subclasses implement terrain generation, scoring objectives, and optional
    hooks for environment configuration or dynamic updates.
    """

    def __init__(self, name: str, description: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Initialises shared level attributes.

        Args:
            name (str): Human-readable level name.
            description (str): Short description displayed in menus or logs.
            metadata (Optional[Dict[str, Any]]): Optional metadata dictionary merged into the level metadata.
        """

        self.name = name
        self.description = description
        self._metadata: Dict[str, Any] = metadata.copy() if metadata else {}
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def generate_terrain(self, space: pymunk.Space):
        """Constructs static bodies (ground, obstacles) in the Pymunk space.

        Args:
            space (pymunk.Space): Pymunk space receiving the generated static geometry.
        """
        pass

    @abstractmethod
    def get_spawn_point(self) -> np.ndarray:
        """Returns the (x, y) coordinates for the lander spawn.

        Returns:
            np.ndarray: Spawn position for the lander.
        """
        pass

    def get_spawn_rotation(self) -> float:
        """Returns the lander's starting angle in radians (default: upright).

        Returns:
            float: Initial rotation angle in radians.
        """
        return 0.0

    def get_initial_velocity(self) -> np.ndarray:
        """Returns an optional initial (vx, vy) velocity vector.

        Returns:
            np.ndarray: Initial velocity vector; defaults to zero.
        """
        return np.zeros(2, dtype=float)

    @abstractmethod
    def check_objectives(self, env) -> dict:
        """Evaluates level-specific completion criteria.

        Args:
            env: Environment instance exposing simulation state used for scoring.

        Returns:
            dict: Mapping of objective names to boolean completion status or metrics.
        """
        pass

    def get_waypoints(self) -> List[np.ndarray]:
        """Returns ordered waypoint coordinates for routes or laps.

        Returns:
            List[np.ndarray]: Collection of waypoint coordinates.
        """
        return []

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, max_x, min_y, max_y) covering the playable area.

        Returns:
            Tuple[float, float, float, float]: Axis-aligned bounding box for the level.
        """
        inf = float("inf")
        return (-inf, inf, -inf, inf)

    def configure_environment(self, env) -> None:
        """Hook for adjusting environment parameters before the episode starts.

        Args:
            env: Environment instance to configure.
        """
        return None

    def reset(self, space: pymunk.Space) -> None:
        """Clears per-episode state; regenerates terrain when necessary.

        Args:
            space (pymunk.Space): Pymunk space supplied for level reuse or regeneration.
        """
        self._state.clear()

    def update(self, dt: float, env=None) -> None:
        """Optional: Handle dynamic level elements (moving platforms).

        Args:
            dt (float): Simulation time step in seconds.
            env: Optional environment reference for reading state.
        """
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """Returns descriptive attributes and author-supplied metadata.

        Returns:
            Dict[str, Any]: Copy of metadata describing the level.
        """
        details = {"name": self.name, "description": self.description}
        details.update(self._metadata)
        return details.copy()
