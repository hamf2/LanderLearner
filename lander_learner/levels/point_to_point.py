from __future__ import annotations

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


class PointToPointLevel(BaseLevel):
    """Spline-driven corridor defined by control points and offset walls.

    The level samples a Catmull-Rom spline to build a centreline, offsets wall
    geometry on each side, and adds semicircular end caps to produce a closed
    collision loop suitable for pymunk.
    """

    def __init__(
        self,
        control_points: Sequence[Iterable[float]],
        corridor_half_width: float = 10.0,
        samples_per_segment: int = 20,
        floor_thickness: float = 0.15,
        wall_thickness: float = 0.1,
        arrival_tolerance: float = 5.0,
        spawn_offset: Tuple[float, float] = (0.0, 6.0),
        description: str = "Spline corridor between two checkpoints.",
        name: str = "Point To Point",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialises the level with spline and corridor parameters.

        Args:
            control_points: Ordered two-dimensional spline control points.
            corridor_half_width: Half-width from centreline to each wall.
            samples_per_segment: Samples per control-point segment for spline evaluation.
            floor_thickness: Thickness for the rendered floor segments.
            wall_thickness: Collision radius applied to pymunk segments.
            arrival_tolerance: Distance threshold in metres for goal completion.
            spawn_offset: Offset from the first control point for spawn placement.
            description: Human-readable level description.
            name: Level name reported through metadata.
            metadata: Optional metadata overrides.
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

        payload = metadata.copy() if metadata else {}
        payload.setdefault("corridor_half_width", corridor_half_width)
        payload.setdefault("arrival_tolerance", arrival_tolerance)
        payload.setdefault("control_point_count", int(control.shape[0]))
        super().__init__(name=name, description=description, metadata=payload)

        self.control_points = control
        self.corridor_half_width = float(corridor_half_width)
        self.samples_per_segment = int(samples_per_segment)
        self.floor_thickness = float(floor_thickness)
        self.wall_thickness = float(wall_thickness)
        self.arrival_tolerance = float(arrival_tolerance)
        self.spawn_offset = np.asarray(spawn_offset, dtype=float)

        self._space: Optional[pymunk.Space] = None
        self._entries: List[_StaticEntry] = []
        self._centreline: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
        self._left_wall: Optional[np.ndarray] = None
        self._right_wall: Optional[np.ndarray] = None
        self._bounds: Optional[Tuple[float, float, float, float]] = None

    def generate_terrain(self, space: pymunk.Space) -> None:
        """Creates all static collision geometry for the level.

        Args:
            space: pymunk space that receives the generated geometry.
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
        """Evaluates goal completion metrics for the supplied environment.

        Args:
            env: Environment exposing a ``lander_position`` attribute.

        Returns:
            Dict[str, Any]: Goal result dictionary containing ``goal_reached``
            and the remaining ``distance_to_goal``.
        """

        final_point = self.control_points[-1]
        position = np.asarray(getattr(env, "lander_position", np.zeros(2)), dtype=float)
        distance = float(np.linalg.norm(position - final_point))
        return {"goal_reached": distance <= self.arrival_tolerance, "distance_to_goal": distance}

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
        """Updates environment attributes (e.g., target position) for this level.

        Args:
            env: Environment instance receiving level-specific configuration.
        """

        if hasattr(env, "target_position"):
            env.target_position = np.asarray(self.control_points[-1], dtype=np.float32)

    def reset(self, space: pymunk.Space) -> None:
        """Rebuilds level geometry for a fresh pymunk space.

        Args:
            space: pymunk space receiving the regenerated static geometry.
        """

        super().reset(space)
        self._space = space
        self._remove_static_geometry()
        self._create_static_geometry()

    def update(self, dt: float, env=None) -> None:
        """No-op hook for dynamic updates; provided for interface completeness.

        Args:
            dt (float): Simulation time step.
            env: Optional environment context provided by the caller.
        """
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """Extends base metadata with point-to-point level identifiers.

        Returns:
            Dict[str, Any]: Metadata dictionary augmented with level type information.
        """
        details = super().get_metadata()
        details.update({"type": "point_to_point"})
        return details

    # --- Internal helpers -------------------------------------------------
    def _compute_geometry(self) -> None:
        """Samples the Catmull–Rom spline and caches geometry derivations."""

        if self._centreline is not None:
            return

        positions, derivatives = self._sample_catmull_rom()
        normals = self._compute_normals(derivatives)

        left_wall = positions + normals * self.corridor_half_width
        right_wall = positions - normals * self.corridor_half_width

        min_xy = np.minimum(np.min(left_wall, axis=0), np.min(right_wall, axis=0))
        max_xy = np.maximum(np.max(left_wall, axis=0), np.max(right_wall, axis=0))
        self._bounds = (float(min_xy[0]), float(max_xy[0]), float(min_xy[1]), float(max_xy[1]))

        self._centreline = positions
        self._normals = normals
        self._left_wall = left_wall
        self._right_wall = right_wall

    def _create_static_geometry(self) -> None:
        """Instantiates pymunk shapes representing the corridor walls."""

        if self._space is None:
            return
        self._remove_static_geometry()

        assert self._centreline is not None and self._left_wall is not None and self._right_wall is not None

        boundary_points = self._generate_boundary_loop()
        if len(boundary_points) < 3:
            return

        loop_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        shapes: List[pymunk.Segment] = []
        total_points = len(boundary_points)
        for idx in range(total_points):
            a = boundary_points[idx]
            b = boundary_points[(idx + 1) % total_points]
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

    def _generate_boundary_loop(self, cap_samples: int = 12) -> List[np.ndarray]:
        """Builds a closed loop polygon for wall segment creation.

        Args:
            cap_samples (int, optional): Number of samples used to approximate
                each semicircular end cap. Defaults to ``12``.

        Returns:
            List[np.ndarray]: Ordered coordinates for the loop.
        """

        left_points = [np.array(pt, dtype=float) for pt in self._left_wall]
        right_points = [np.array(pt, dtype=float) for pt in self._right_wall[::-1]]

        forward_cap = self._generate_endcap_points(len(self._centreline) - 1, True, cap_samples)
        backward_cap = self._generate_endcap_points(0, False, cap_samples)

        boundary: List[np.ndarray] = []
        boundary.extend(left_points)
        boundary.extend(forward_cap)
        boundary.extend(right_points[1:])
        boundary.extend(backward_cap[:-1])
        return boundary

    def _generate_endcap_points(self, idx: int, forward: bool, samples: int) -> List[np.ndarray]:
        """Interpolates semicircular endcap points.

        Args:
            idx (int): Index into the pre-sampled spline arrays.
            forward (bool): ``True`` for the terminal end, ``False`` for the origin.
            samples (int): Number of angular samples to generate.

        Returns:
            List[np.ndarray]: Interpolated points for the requested cap.
        """

        center = self._centreline[idx]
        normal = self._normals[idx]
        tangent = np.array([normal[1], -normal[0]])
        radius = self.corridor_half_width

        start_angle = 0 if forward else np.pi
        end_angle = np.pi if forward else 0
        angles = np.linspace(start_angle, end_angle, num=max(samples, 2), endpoint=True)[1:]

        if not forward:
            tangent *= -1.0

        points: List[np.ndarray] = []
        for theta in angles[:-1]:
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            direction = normal * cos_theta + tangent * sin_theta
            points.append(center + direction * radius)

        end_point = self._right_wall[idx] if forward else self._left_wall[idx]
        points.append(np.array(end_point, dtype=float))
        return points

    def _sample_catmull_rom(self) -> Tuple[np.ndarray, np.ndarray]:
        """Samples the Catmull–Rom spline and its first derivative.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sampled positions and derivative vectors.
        """

        pts = self.control_points
        padded = np.vstack([pts[0], pts, pts[-1]])
        samples: List[np.ndarray] = []
        derivatives: List[np.ndarray] = []
        seg_samples = self.samples_per_segment

        for idx in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[idx - 1], padded[idx], padded[idx + 1], padded[idx + 2]
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
        samples.append(pts[-1])
        derivatives.append(pts[-1] - pts[-2])

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


class PointToPointPresetLevel(PointToPointLevel):
    """Factory-backed point-to-point level driven by JSON presets."""

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
        name = merged.pop("name", f"PointToPoint-{preset_name}")
        description = merged.pop("description", "Spline corridor preset")

        metadata = merged.pop("metadata", {})
        if override_metadata:
            metadata = {**metadata, **override_metadata}

        super().__init__(
            control_points=control_points,
            description=description,
            metadata={"preset": preset_name, **metadata, "label": name},
            **merged,
        )
