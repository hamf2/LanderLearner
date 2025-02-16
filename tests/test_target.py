import numpy as np
import pytest
from lander_learner.utils.target import TargetZone

def test_target_zone_reset_and_initial_position():
    tz = TargetZone()
    tz.reset()
    pos = tz.initial_position
    # Check that initial_position is a 2-element numpy array.
    assert isinstance(pos, np.ndarray)
    assert pos.shape[0] == 2

def test_target_zone_get_position_static():
    tz = TargetZone()
    # Disable motion to test static behavior.
    tz.motion_enabled = False
    tz.reset()
    pos1 = tz.get_target_position(elapsed_time=0)
    pos2 = tz.get_target_position(elapsed_time=100)
    # With motion disabled, the position should remain constant.
    assert np.array_equal(pos1, pos2)

def test_target_zone_get_position_motion():
    tz = TargetZone(motion_enabled=True, vel_range_x=5)
    tz.motion_interval = 1.0
    tz.reset()
    non_zero_velocity = False
    for i in range(5):
        pos_initial = tz.get_target_position(elapsed_time=np.array([0.0 + i]))
        pos_after = tz.get_target_position(elapsed_time=np.array([1.0 + i]))
        if not np.array_equal(tz.current_velocity, np.array([0.0, 0.0])):
            # With motion enabled, the position should change after a full motion segment.
            assert not np.array_equal(pos_initial, pos_after)
            non_zero_velocity = True
            break
    assert non_zero_velocity, "Target position did not change after multiple motion segments."
    
def test_target_zone_get_position_on_ground():
    tz = TargetZone(spawn_mode = "on_ground", motion_enabled=False)
    tz.reset()
    assert tz.initial_position[1] == 0.0, "Target did not spawn on the ground."
