from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pymunk

from .base_level import BaseLevel


class BlankLevel(BaseLevel):
    """An empty, infinite level with no obstacles.

    This level provides a minimal environment useful for debugging or
    running the agent without any terrain. The ground is intentionally
    omitted; collision-based landing can still be detected by external
    fixtures should they be added to the space by other code.
    """

    def __init__(
        self,
        name: str = "Blank",
        description: str = "Empty infinite level with no obstacles.",
        metadata: Optional[Dict[str, Any]] = None,
        target_zone_kwargs: Optional[Dict[str, Any]] = None,
        episode_time_limit: Optional[float] = None,
        initial_fuel: Optional[float] = None,
    ) -> None:
        payload = metadata.copy() if metadata else {}
        super().__init__(
            name=name,
            description=description,
            metadata=payload,
            target_zone_kwargs=target_zone_kwargs,
            episode_time_limit=episode_time_limit,
            initial_fuel=initial_fuel,
        )

    def generate_terrain(self, space: pymunk.Space) -> None:
        """No terrain is added for the blank level."""
        return None

    def get_spawn_point(self) -> np.ndarray:
        """Return a sensible default spawn point above the origin."""
        return np.array([0.0, 12.0], dtype=float)

    def check_objectives(self, env) -> Dict[str, bool]:
        """Evaluate simple landing objective using collision_state.

        Returns a dict with keys ``landed`` and ``within_bounds``. The
        blank level treats the world as unbounded so ``within_bounds`` is
        always True.
        """

        landed = bool(getattr(env, "collision_state", False))
        return {"landed": landed, "within_bounds": True}

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Return infinite bounds for the blank level."""

        return (float("-inf"), float("inf"), float("-inf"), float("inf"))

    def configure_environment(self, env) -> None:
        """No special environment configuration for blank level."""
        return None

    def reset(self, space: pymunk.Space) -> None:
        """Reset per-episode state (no terrain to recreate)."""
        super().reset(space)
        return None


__all__ = ["BlankLevel"]
