import pathlib


class Config:
    # Physics / Environment
    GRAVITY = 1.0
    TIME_STEP = 1.0 / 60.0
    THRUST_POWER = 40.0  # Max force from each thruster
    FUEL_COST = 0.003  # Fuel cost per Newton of thrust
    INITIAL_FUEL = 100.0  # Starting fuel
    IMPULSE_THRESHOLD = 30.0  # Max impulse before crash (Newton-seconds)
    IDLE_TIMEOUT = 3.0  # Max time lander can be idle before episode termination (seconds)
    MAX_EPISODE_DURATION = 20.0  # Max duration of an episode (seconds)

    # Lander dimensions
    LANDER_WIDTH = 3.0
    LANDER_HEIGHT = 1.0
    LANDER_MASS = 10.0
    LANDER_COF = 1.0

    # Target zone (if target_zone_mode is True in environment)
    # Target zone is a rectangular area where the lander is incentivised to travel/land
    # Target zone is defined by its center position, width, and height
    # Spawn mode options are "deterministic", "on_ground", "above_ground"
    TARGET_ZONE_SPAWN_MODE = "on_ground"
    TARGET_ZONE_SPAWN_RANGE_X = 60.0
    TARGET_ZONE_SPAWN_RANGE_Y = 50.0
    TARGET_ZONE_X = 30.0
    TARGET_ZONE_Y = 0.0
    TARGET_ZONE_WIDTH = 10.0
    TARGET_ZONE_HEIGHT = 5.0
    TARGET_ZONE_MOTION = False  # If True, target zone moves randomly with piecewise linear motion
    TARGET_ZONE_MOTION_INTERVAL = 5.0
    TARGET_ZONE_VELOCITY_RANGE_X = 5.0
    TARGET_ZONE_VELOCITY_RANGE_Y = 2.0
    REQUIRED_LAPS = 3  # Number of laps required to complete episode in lap levels

    # Sensor specifications
    LASER_RANGE = 100.0

    # Rendering
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    REPLAY_SPEED = 1.0
    RENDER_SCALE = 10.0

    # Calculated
    PHYSICS_STEPS_PER_FRAME = int(1.0 / TIME_STEP / FPS)
    FRAME_TIME_STEP = 1 / FPS

    # GUI recordings directory
    DEFAULT_RECORDING_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "recordings"


class RL_Config:
    DEFAULT_CHECKPOINT_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "checkpoints"
    DEFAULT_LOGGING_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "logs"
    CHECKPOINT_FREQ = 500000

    # PPO Configuration
    PPO_OPTIONS = {
        "device": "cpu"
    }

    # SAC Configuration
    SAC_OPTIONS = {
        "device": "auto"
    }

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
