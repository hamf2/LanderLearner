from __future__ import annotations

"""Closed-loop spline level that tracks lap completions across a start line."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pymunk

from .base_level import BaseLevel
from .level_data import load_level_payload


@dataclass
class _StaticEntry:
    body: pymunk.Body
    shapes: Tuple[pymunk.Shape, ...]


class LapLevel(BaseLevel):
    """Spline corridor that forms a loop and tracks lap completions.

    The level samples a closed Catmull--Rom spline to build its centreline,
    offsets wall geometry on both sides, and monitors crossings of the start
    line so environments can count completed laps.
    """

    def __init__(
        self,
        control_points: Sequence[Iterable[float]],
        corridor_half_width: float = 10.0,
        samples_per_segment: int = 20,
        wall_thickness: float = 0.1,
        target_laps: int = 3,
        spawn_offset: Tuple[float, float] = (0.0, 6.0),
        description: str = "Closed spline corridor with lap tracking.",
        name: str = "Lap Course",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialises the lap level with spline parameters and lap goals.

        Args:
            control_points: Ordered two-dimensional spline control points that
                define the closed loop. The first and last points need not match;
                the spline wraps automatically.
            corridor_half_width: Half-width from centreline to each wall.
            samples_per_segment: Samples per control-point segment for spline evaluation.
            wall_thickness: Collision radius applied to pymunk segments.
            target_laps: Number of laps required for goal completion.
            spawn_offset: Offset applied to the first control point for spawn placement.
            description: Human-readable level description.
            name: Level name reported through metadata.
            metadata: Optional metadata overrides merged into level metadata.
        """

        control = np.asarray(control_points, dtype=float)
        if control.ndim != 2 or control.shape[1] != 2:
            raise ValueError("control_points must be an iterable of 2D coordinates")
        if control.shape[0] < 4:
            raise ValueError("At least four control points are required for spline generation")
        if corridor_half_width <= 0.0:
            raise ValueError("corridor_half_width must be positive")
        if samples_per_segment < 4:
            raise ValueError("samples_per_segment must be at least four for smooth interpolation")
        if target_laps < 1:
            raise ValueError("target_laps must be at least one")

        payload = metadata.copy() if metadata else {}
        payload.setdefault("corridor_half_width", corridor_half_width)
        payload.setdefault("target_laps", target_laps)
        payload.setdefault("control_point_count", int(control.shape[0]))
        super().__init__(name=name, description=description, metadata=payload)

        self.control_points = control
        self.corridor_half_width = float(corridor_half_width)
        self.samples_per_segment = int(samples_per_segment)
        self.wall_thickness = float(wall_thickness)
        self.target_laps = int(target_laps)
        self.spawn_offset = np.asarray(spawn_offset, dtype=float)

        self._space: Optional[pymunk.Space] = None
        self._entries: List[_StaticEntry] = []
        self._centreline: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
        self._left_wall: Optional[np.ndarray] = None
        self._right_wall: Optional[np.ndarray] = None
        self._bounds: Optional[Tuple[float, float, float, float]] = None
        self._start_point: Optional[np.ndarray] = None
        self._start_direction: Optional[np.ndarray] = None

    def generate_terrain(self, space: pymunk.Space) -> None:
        """Creates all static collision geometry for the level.

        Args:
            space: Pymunk space that receives the generated geometry.
        """

        self._space = space
        self._compute_geometry()
        self._create_static_geometry()

    def get_spawn_point(self) -> np.ndarray:
        """Returns the spawn position derived from the first control point.

        Returns:
            np.ndarray: Spawn coordinates offset from the initial control point.
        """

        start = self.control_points[0]
        return np.asarray(start + self.spawn_offset, dtype=float)

    def check_objectives(self, env) -> Dict[str, Any]:
        """Reports laps completed and whether the target has been achieved.

        Args:
            env: Environment exposing a ``lander_position`` attribute (optional).

        Returns:
            Dict[str, Any]: Dictionary containing ``laps_completed`` and
            ``goal_reached`` fields.
        """

        laps_completed = int(self._state.get("laps_completed", 0))
        return {"laps_completed": laps_completed, "goal_reached": laps_completed >= self.target_laps}

    def get_waypoints(self) -> List[np.ndarray]:
        """Returns a copy of the spline control points as navigational waypoints.

        Returns:
            List[np.ndarray]: Independent copies of each control point.
        """

        return [cp.copy() for cp in self.control_points]

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Returns the level bounds encompassing both walls.

        Returns:
            Tuple[float, float, float, float]: Min/max x and y limits of the geometry.
        """

        if self._bounds is None:
            self._compute_geometry()
        assert self._bounds is not None
        return self._bounds

    def configure_environment(self, env) -> None:
        """Updates environment attributes (e.g., target laps) for this level.

        Args:
            env: Environment instance receiving level-specific configuration.
        """

        if hasattr(env, "target_laps"):
            env.target_laps = self.target_laps
        if hasattr(env, "laps_completed"):
            env.laps_completed = int(self._state.get("laps_completed", 0))

    def get_finish_line(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Returns endpoints spanning the corridor width at the start line."""

        if self._start_point is None or self._start_direction is None:
            self._compute_geometry()
        if self._start_point is None or self._start_direction is None:
            return None

        normal = np.array([-self._start_direction[1], self._start_direction[0]], dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            normal = np.array([0.0, 1.0], dtype=float)
        else:
            normal /= norm

        left = self._start_point + normal * self.corridor_half_width
        right = self._start_point - normal * self.corridor_half_width
        return left, right

    def reset(self, space: pymunk.Space) -> None:
        """Rebuilds level geometry for a fresh pymunk space.

        Args:
            space: Pymunk space receiving the regenerated static geometry.
        """

        super().reset(space)
        self._space = space
        self._remove_static_geometry()
        self._create_static_geometry()
        self._state["laps_completed"] = 0
        self._state["last_signed_distance"] = None
        self._state["lap_armed"] = False

    def update(self, dt: float, env=None) -> None:
        """Tracks laps by monitoring crossings of the start line.

        Args:
            dt (float): Simulation time step (unused).
            env: Optional environment context providing the lander position.
        """

        lander_pos = getattr(env, "lander_position", None)
        if lander_pos is None:
            return

        position = np.asarray(lander_pos, dtype=float)
        signed_distance = float(np.dot(position - self._start_point, self._start_direction))

        last_signed = self._state.get("last_signed_distance")
        lap_armed = bool(self._state.get("lap_armed", False))

        if last_signed is None:
            self._state["last_signed_distance"] = signed_distance
            self._state["lap_armed"] = signed_distance <= 0.0
            if hasattr(env, "laps_completed"):
                env.laps_completed = int(self._state.get("laps_completed", 0))
            return

        if not lap_armed and signed_distance <= 0.0:
            lap_armed = True

        if lap_armed and last_signed < 0.0 and signed_distance >= 0.0:
            self._state["laps_completed"] = int(self._state.get("laps_completed", 0)) + 1
            lap_armed = False

        self._state["lap_armed"] = lap_armed
        self._state["last_signed_distance"] = signed_distance

        if hasattr(env, "laps_completed"):
            env.laps_completed = int(self._state.get("laps_completed", 0))

    def get_metadata(self) -> Dict[str, Any]:
        """Extends base metadata with lap level identifiers.

        Returns:
            Dict[str, Any]: Metadata dictionary augmented with level type information.
        """

        details = super().get_metadata()
        details.update({"type": "lap", "target_laps": self.target_laps})
        return details

    # --- Internal helpers -------------------------------------------------
    def _compute_geometry(self) -> None:
        """Samples the closed Catmull--Rom spline and caches geometry derivations."""

        if self._centreline is not None:
            return

        positions, derivatives = self._sample_catmull_rom()
        normals = self._compute_normals(derivatives)

        left_wall = positions + normals * self.corridor_half_width
        right_wall = positions - normals * self.corridor_half_width

        reference = self.control_points[0]
        start_idx = int(np.argmin(np.linalg.norm(positions - reference, axis=1)))
        if start_idx != 0:
            positions = np.roll(positions, -start_idx, axis=0)
            derivatives = np.roll(derivatives, -start_idx, axis=0)
            normals = np.roll(normals, -start_idx, axis=0)
            left_wall = np.roll(left_wall, -start_idx, axis=0)
            right_wall = np.roll(right_wall, -start_idx, axis=0)

        min_xy = np.minimum(np.min(left_wall, axis=0), np.min(right_wall, axis=0))
        max_xy = np.maximum(np.max(left_wall, axis=0), np.max(right_wall, axis=0))
        self._bounds = (float(min_xy[0]), float(max_xy[0]), float(min_xy[1]), float(max_xy[1]))

        self._centreline = positions
        self._normals = normals
        self._left_wall = left_wall
        self._right_wall = right_wall

        tangent = derivatives[0]
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            tangent = np.array([1.0, 0.0])
        else:
            tangent = tangent / norm
        self._start_point = positions[0].copy()
        self._start_direction = tangent

    def _create_static_geometry(self) -> None:
        """Instantiates pymunk shapes representing the corridor walls."""

        if self._space is None:
            return
        self._remove_static_geometry()

        assert self._centreline is not None and self._left_wall is not None and self._right_wall is not None

        loop_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        shapes: List[pymunk.Segment] = []

        for points in (self._left_wall, self._right_wall):
            total = len(points)
            if total < 2:
                return
            for idx in range(total):
                a = points[idx]
                b = points[(idx + 1) % total]
                if np.allclose(a, b):
                    continue
                segment = pymunk.Segment(loop_body, tuple(a), tuple(b), self.wall_thickness)
                segment.friction = 1.0
                segment.elasticity = 0.0
                shapes.append(segment)

        if not shapes:
            return

        self._entries = [_StaticEntry(loop_body, tuple(shapes))]
        self._space.add(loop_body, *shapes)

    def _remove_static_geometry(self) -> None:
        """Detaches previously created static geometry from the space."""

        if self._space is None:
            return
        if not self._entries:
            return
        for entry in self._entries:
            try:
                self._space.remove(entry.body, *entry.shapes)
            except Exception:
                continue
        self._entries.clear()

    def _sample_catmull_rom(self) -> Tuple[np.ndarray, np.ndarray]:
        """Samples the closed Catmull--Rom spline and its first derivative."""

        pts = self.control_points
        padded = np.vstack([pts[-2:], pts, pts[:2]])
        samples: List[np.ndarray] = []
        derivatives: List[np.ndarray] = []
        seg_samples = self.samples_per_segment
        segments = len(pts)

        for idx in range(1, segments + 1):
            p0, p1, p2, p3 = (
                padded[idx - 1],
                padded[idx],
                padded[idx + 1],
                padded[idx + 2],
            )
            for j in range(seg_samples):
                t = j / seg_samples
                pos = 0.5 * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
                )
                deriv = 0.5 * (
                    (-p0 + p2)
                    + 2 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * t
                    + 3 * (-p0 + 3 * p1 - 3 * p2 + p3) * t * t
                )
                samples.append(pos)
                derivatives.append(deriv)
        samples.append(samples[0])
        derivatives.append(derivatives[0])

        return np.asarray(samples, dtype=float), np.asarray(derivatives, dtype=float)

    def _compute_normals(self, derivatives: np.ndarray) -> np.ndarray:
        """Normalises derivative vectors and returns perpendicular normals.

        Args:
            derivatives (np.ndarray): First-derivative vectors aligned with the spline samples.

        Returns:
            np.ndarray: Normal vectors corresponding to each derivative.
        """

        normals = np.zeros_like(derivatives)
        previous = np.array([0.0, 1.0])
        for idx, deriv in enumerate(derivatives):
            norm = np.linalg.norm(deriv)
            if norm < 1e-8:
                normals[idx] = previous
                continue
            tangent = deriv / norm
            normal = np.array([-tangent[1], tangent[0]])
            previous = normal
            normals[idx] = normal
        return normals


class LapPresetLevel(LapLevel):
    """Factory-backed lap level driven by JSON presets."""

    def __init__(self, preset_name: str, override_metadata: Optional[Dict[str, Any]] = None, **overrides: Any) -> None:
        """Constructs the level using values loaded from preset data.

        Args:
            preset_name: Name of the JSON preset bundled with the package.
            override_metadata: Optional metadata overrides merged with preset values.
            **overrides: Keyword overrides applied to preset parameters.
        """

        payload = load_level_payload(preset_name)

        control_points = overrides.pop("control_points", payload.pop("control_points"))
        payload.pop("type", None)

        merged: Dict[str, Any] = {**payload, **overrides}
        name = merged.pop("name", f"Lap-{preset_name}")
        description = merged.pop("description", "Closed spline lap preset")

        metadata = merged.pop("metadata", {})
        if override_metadata:
            metadata = {**metadata, **override_metadata}

        super().__init__(
            control_points=control_points,
            description=description,
            metadata={"preset": preset_name, **metadata, "label": name},
            **merged,
        )
