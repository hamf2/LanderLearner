import numpy as np
import pymunk
import pytest

from lander_learner.levels import (
    HalfPlaneLevel,
    LapLevel,
    LapPresetLevel,
    PointToPointLevel,
    PointToPointPresetLevel,
    get_level,
)
from lander_learner.physics import PhysicsEngine
from lander_learner.utils.dataclasses import BodySegment


@pytest.fixture
def pymunk_space():
    space = pymunk.Space()
    space.gravity = (0.0, -9.81)
    return space


def test_half_plane_initialisation_and_alignment(pymunk_space):
    level = HalfPlaneLevel(plane_width=150.0, surface_friction=0.8)
    level.generate_terrain(pymunk_space)

    assert level.get_spawn_point().shape == (2,)
    assert level.get_bounds() == (-float("inf"), float("inf"), 0.0, float("inf"))

    level.reset(pymunk_space)
    half_width = level.plane_width / 2.0
    assert half_width == 75.0
    assert len(pymunk_space.shapes) == 1

    segment = next(iter(pymunk_space.shapes))
    assert isinstance(segment, pymunk.Segment)
    assert np.isclose(segment.a.x, -half_width, atol=1e-3)
    assert np.isclose(segment.b.x, half_width, atol=1e-3)
    assert np.isclose(segment.a.y, 0.0, atol=1e-6)
    assert np.isclose(segment.b.y, 0.0, atol=1e-6)

    class EnvStub:
        def __init__(self, x):
            self.lander_position = np.array([x, 12.0])

    env = EnvStub(42.0)
    level.update(1 / 60, env=env)

    segment_after = next(iter(pymunk_space.shapes))
    assert np.isclose(level._ground_body.position.x, env.lander_position[0], atol=1e-6)
    assert np.isclose(segment_after.a.y, 0.0, atol=1e-6)
    assert np.isclose(segment_after.b.y, 0.0, atol=1e-6)


def test_point_to_point_initialisation_and_geometry(pymunk_space):
    control_points = [
        (-25.0, 10.0),
        (-5.0, 14.0),
        (15.0, 12.0),
        (35.0, 8.0),
        (55.0, 8.0),
    ]
    level = PointToPointLevel(control_points=control_points, corridor_half_width=6.0, samples_per_segment=12)
    level.generate_terrain(pymunk_space)

    centreline = level.get_waypoints()
    assert len(centreline) == len(control_points)

    kwargs = level.get_target_zone_kwargs()
    assert kwargs is not None
    assert kwargs["spawn_mode"] == "deterministic"
    assert np.isclose(kwargs["deterministic_x"], control_points[-1][0])
    assert np.isclose(kwargs["deterministic_y"], control_points[-1][1])
    assert np.isclose(kwargs["zone_width"], 12.0)
    assert np.isclose(kwargs["zone_height"], 6.0)

    level.reset(pymunk_space)
    left_points = level._left_wall
    right_points = level._right_wall
    assert left_points is not None and right_points is not None

    samples = len(left_points)
    for idx in range(min(samples, 5)):
        offset = left_points[idx] - right_points[idx]
        assert np.isclose(np.linalg.norm(offset), 12.0, atol=2.0)

    offsets = np.diff(left_points, axis=0)
    assert np.all(np.linalg.norm(offsets, axis=1) > 0.0)


def test_point_to_point_preset_initialisation(pymunk_space):
    level = PointToPointPresetLevel("p2p001", spawn_offset=(0.0, 10.0))
    level.generate_terrain(pymunk_space)

    metadata = level.get_metadata()
    assert metadata["preset"] == "p2p001"
    assert metadata["label"].startswith("PointToPoint")

    kwargs = level.get_target_zone_kwargs()
    assert kwargs is not None
    assert kwargs["spawn_mode"] == "deterministic"
    final_point = level.control_points[-1]
    assert np.isclose(kwargs["deterministic_x"], final_point[0])
    assert np.isclose(kwargs["deterministic_y"], final_point[1])

    level.reset(pymunk_space)
    assert level.control_points.shape[0] >= 4


def test_body_vertices_half_plane_filters():
    engine = PhysicsEngine(level=HalfPlaneLevel(plane_width=120.0))

    empty_geometry = engine.get_body_vertices()
    assert not empty_geometry.segments and not empty_geometry.polys

    kinematic_geometry = engine.get_body_vertices(kinematic=True)
    assert len(kinematic_geometry.segments) == 1
    segment = kinematic_geometry.segments[0]
    assert isinstance(segment, BodySegment)
    assert segment.body_type == "kinematic"
    assert not kinematic_geometry.polys

    lander_geometry = engine.get_body_vertices(lander=True)
    assert lander_geometry.polys and not lander_geometry.segments
    assert all(poly.body_type == "lander" for poly in lander_geometry.polys)


def test_body_vertices_point_to_point_static_geometry():
    control_points = [(-20.0, 10.0), (-5.0, 14.0), (15.0, 12.0), (30.0, 8.0)]
    engine = PhysicsEngine(
        level=PointToPointLevel(control_points=control_points, corridor_half_width=5.0, samples_per_segment=8)
    )

    geometry = engine.get_body_vertices(kinematic=True)
    assert geometry.segments
    assert all(seg.body_type == "kinematic" for seg in geometry.segments)
    assert not geometry.polys


def test_half_plane_ground_target_preset():
    level = get_level("hp_ground_target")
    assert isinstance(level, HalfPlaneLevel)

    kwargs = level.get_target_zone_kwargs()
    assert kwargs is not None
    assert kwargs["spawn_mode"] == "on_ground"
    assert kwargs["motion_enabled"] is False
    assert kwargs["spawn_range_x"] == 80.0


def test_half_plane_moving_target_preset():
    level = get_level("hp_moving_target")
    assert isinstance(level, HalfPlaneLevel)

    kwargs = level.get_target_zone_kwargs()
    assert kwargs is not None
    assert kwargs["spawn_mode"] == "above_ground"
    assert kwargs["motion_enabled"] is True
    assert kwargs["vel_range_y"] == 3.0


def test_lap_level_lap_tracking(pymunk_space):
    control_points = [
        (0.0, 0.0),
        (40.0, 10.0),
        (55.0, 35.0),
        (20.0, 65.0),
        (-30.0, 45.0),
        (-45.0, 15.0),
    ]
    level = LapLevel(control_points=control_points, corridor_half_width=8.0, samples_per_segment=16, target_laps=2)
    level.generate_terrain(pymunk_space)
    level.reset(pymunk_space)

    assert level.get_metadata()["type"] == "lap"
    assert len(level.get_waypoints()) == len(control_points)

    class EnvStub:
        def __init__(self, start_point, start_dir):
            self.lander_position = start_point - start_dir * 5.0
            self.laps_completed = 0
            self.target_laps = 0

    start_point = level._start_point.copy()  # type: ignore[attr-defined]
    start_dir = level._start_direction.copy()  # type: ignore[attr-defined]
    env = EnvStub(start_point, start_dir)

    level.update(1 / 60, env=env)
    env.lander_position = start_point + start_dir * 2.0
    level.update(1 / 60, env=env)

    env.lander_position = start_point - start_dir * 3.0
    level.update(1 / 60, env=env)
    env.lander_position = start_point + start_dir * 4.0
    level.update(1 / 60, env=env)

    objectives = level.check_objectives(env)
    assert objectives["laps_completed"] == 2
    assert objectives["goal_reached"] is True


def test_lap_preset_metadata_and_geometry(pymunk_space):
    level = LapPresetLevel("lap001", spawn_offset=(0.0, 6.0))
    level.generate_terrain(pymunk_space)
    level.reset(pymunk_space)

    metadata = level.get_metadata()
    assert metadata["preset"] == "lap001"
    assert metadata["type"] == "lap"
    assert metadata["target_laps"] >= 1

    geometry = PhysicsEngine(level=level).get_body_vertices(kinematic=True)
    assert geometry.segments
