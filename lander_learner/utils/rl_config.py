import pathlib


class RL_Config:
    DEFAULT_CHECKPOINT_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "checkpoints"
    DEFAULT_LOGGING_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "logs"
    CHECKPOINT_FREQ = 500000

    # PPO Configuration
    PPO_DEVICE = "cpu"

    # SAC Configuration
    SAC_DEVICE = "auto"

    # Default parameters for the DefaultReward
    DEFAULT_DEFAULT_REWARD_PARAMS = {
        "x_velocity_factor": 10.0,
        "angle_penalty_factor": 1.0,
        "collision_penalty": 5.0,
        "crash_penalty_multiplier": 1.0,
    }

    # Default parameters for the RightwardReward
    DEFAULT_RIGHTWARD_REWARD_PARAMS = {
        "x_velocity_factor": 10.0,
        "angle_penalty_factor": 1.0,
        "collision_penalty": 5.0,
        "crash_penalty_multiplier": 1.0,
    }

    # Default parameters for the SoftLandingReward
    DEFAULT_SOFT_LANDING_REWARD_PARAMS = {
        "on_target_touch_down_bonus": 10.0,
        "off_target_touch_down_penalty": 5.0,
        "on_target_idle_bonus": 20.0,
        "off_target_idle_penalty": 2.0,
        "crash_penalty_multiplier": 1.0,
        "time_penalty_factor": 1.0,
        "travel_reward_factor": 3.0,
        "near_target_off_angle_penalty": 3.0,
        "near_target_high_velocity_penalty": 3.0,
        "near_target_high_velocity_cut_off": 1.0,
        "near_target_unit_dist": 5.0,
        "near_target_max_multiplier": 2.5,
        "near_target_passive_bonus": 1.0
    }
