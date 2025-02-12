import pathlib

class Config:
    SCENARIO_FILE = pathlib.Path(__file__).parent.parent / "scenarios" / "scenarios.json"
    DEFAULT_CHECKPOINT_DIR = pathlib.Path(__file__).parent.parent / "agents" / "checkpoints"
    DEFAULT_LOGGING_DIR = pathlib.Path(__file__).parent.parent / "agents" / "logs"
    
    # Physics / Environment
    GRAVITY = 1.0
    TIME_STEP = 1.0 / 60.0
    THRUST_POWER = 40.0         # Max force from each thruster
    FUEL_COST = 0.005           # Fuel cost per Newton of thrust
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
    TARGET_ZONE_X = 30.0
    TARGET_ZONE_Y = 0.0
    TARGET_ZONE_WIDTH = 10.0
    TARGET_ZONE_HEIGHT = 5.0

    # Sensor specifications
    LASER_RANGE = 100.0

    # Rendering
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    REPLAY_SPEED = 1.0
    RENDER_SCALE = 10.0

    # RL

    # Calculated
    PHYSICS_STEPS_PER_FRAME = int(1.0 / TIME_STEP / FPS)
    RENDER_TIME_STEP = 1 / FPS
