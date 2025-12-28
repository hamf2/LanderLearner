"""Shared dataclass definitions for geometry exported by physics utilities.

These structures abstract pymunk shapes into serialisable primitives that can
be reused by rendering layers or logging utilities without depending on pymunk
itself.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BodySegment:
    """Segment primitive exported by ``get_body_vertices``.

    Attributes:
        start (Tuple[float, float]): World-space start coordinate of the segment.
        end (Tuple[float, float]): World-space end coordinate of the segment.
        radius (float): Segment radius as defined in pymunk.
        body_type (str): Category of the owning body (static, kinematic, etc.).
    """

    start: Tuple[float, float]
    end: Tuple[float, float]
    radius: float
    body_type: str


@dataclass
class BodyPolygon:
    """Polygon primitive exported by ``get_body_vertices``.

    Attributes:
        vertices (List[Tuple[float, float]]): Polygon vertices in world space.
        body_type (str): Category of the owning body (static, kinematic, etc.).
    """

    vertices: List[Tuple[float, float]]
    body_type: str


@dataclass
class BodyGeometry:
    """Container returned by ``get_body_vertices`` holding terrain primitives.

    Attributes:
        segments (List[BodySegment]): Segment primitives associated with the
            requested bodies.
        polys (List[BodyPolygon]): Polygon primitives associated with the
            requested bodies.
    """

    segments: List[BodySegment]
    polys: List[BodyPolygon]


__all__ = ["BodySegment", "BodyPolygon", "BodyGeometry"]
