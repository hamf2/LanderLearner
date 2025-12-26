from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pymunk

from .base_level import BaseLevel


class HalfPlaneLevel(BaseLevel):
    """Infinite half-plane that keeps the landing surface beneath the vehicle."""

    def __init__(
        self,
        plane_width: float = 200.0,
        surface_friction: float = 1.0,
        surface_elasticity: float = 0.5,
        description: str = "Kinematic plane that recenters under the lander.",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = metadata.copy() if metadata else {}
        payload.setdefault("plane_width", plane_width)
        payload.setdefault("surface_friction", surface_friction)
        payload.setdefault("surface_elasticity", surface_elasticity)
        super().__init__(name="Half Plane", description=description, metadata=payload)

        self.plane_width = plane_width
        self.surface_friction = surface_friction
        self.surface_elasticity = surface_elasticity

        self._space: Optional[pymunk.Space] = None
        self._ground_body: Optional[pymunk.Body] = None
        self._ground_shape: Optional[pymunk.Segment] = None
        self._ground_in_space = False

    def generate_terrain(self, space: pymunk.Space) -> None:
        self._space = space
        self._ground_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._ground_body.position = (0.0, 0.0)

        half_width = self.plane_width / 2.0
        self._ground_shape = pymunk.Segment(
            self._ground_body,
            (-half_width, 0.0),
            (half_width, 0.0),
            0.1,
        )
        self._ground_shape.friction = self.surface_friction
        self._ground_shape.elasticity = self.surface_elasticity

        space.add(self._ground_body, self._ground_shape)
        self._ground_in_space = True

    def get_spawn_point(self) -> np.ndarray:
        return np.array([0.0, 12.0], dtype=float)

    def check_objectives(self, env) -> Dict[str, bool]:
        position = np.array(getattr(env, "lander_position", (0.0, 0.0)), dtype=float)
        landed = bool(getattr(env, "collision_state", False))
        within_bounds = bool(abs(position[0]) <= self.plane_width / 2.0)
        return {"landed": landed, "within_bounds": within_bounds}

    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-self.plane_width / 2.0, self.plane_width / 2.0, 0.0, float("inf"))

    def configure_environment(self, env) -> None:
        return None

    def reset(self, space: pymunk.Space) -> None:
        super().reset(space)
        self._space = space
        if self._ground_body is None or self._ground_shape is None:
            return

        if self._ground_in_space:
            space.remove(self._ground_body, self._ground_shape)
            self._ground_in_space = False

        self._ground_body.position = (0.0, 0.0)
        space.add(self._ground_body, self._ground_shape)
        self._ground_in_space = True

    def update(self, dt: float, env=None) -> None:
        if env is None or self._ground_body is None or self._ground_shape is None:
            return

        lander_pos = getattr(env, "lander_position", None)
        if lander_pos is None:
            return

        target_x = float(lander_pos[0])
        if abs(self._ground_body.position.x - target_x) < 1e-6:
            return

        if self._ground_in_space and self._space is not None:
            self._space.remove(self._ground_body, self._ground_shape)
            self._ground_in_space = False

        self._ground_body.position = (target_x, 0.0)

        if self._space is not None:
            self._space.add(self._ground_body, self._ground_shape)
            self._ground_in_space = True

    def get_metadata(self) -> Dict[str, Any]:
        details = super().get_metadata()
        details.update({"type": "half_plane"})
        return details
