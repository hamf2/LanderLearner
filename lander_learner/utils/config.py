import pathlib

class Config:
    DEFAULT_CHECKPOINT_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "checkpoints"
    DEFAULT_LOGGING_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "logs"
    
    # Physics / Environment
    GRAVITY = 1.0
    TIME_STEP = 1.0 / 60.0
    THRUST_POWER = 40.0         # Max force from each thruster
    FUEL_COST = 0.003           # Fuel cost per Newton of thrust
    INITIAL_FUEL = 100.0        # Starting fuel
    IMPULSE_THRESHOLD = 30.0    # Max impulse before crash (Newton-seconds)
    IDLE_TIMEOUT = 3.0          # Max time lander can be idle before episode termination (seconds)
    MAX_EPISODE_DURATION = 20.0 # Max duration of an episode (seconds)

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
    TARGET_ZONE_MOTION = False # If True, target zone moves randomly with piecewise linear motion
    TARGET_ZONE_MOTION_INTERVAL = 5.0
    TARGET_ZONE_VELOCITY_RANGE_X = 5.0
    TARGET_ZONE_VELOCITY_RANGE_Y = 2.0

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
    RENDER_TIME_STEP = 1 / FPS
