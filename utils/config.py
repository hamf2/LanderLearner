import pathlib

class Config:
    SCENARIO_FILE = pathlib.Path(__file__).parent.parent / "scenarios" / "scenarios.json"
    
    # Physics / Environment
    GRAVITY = 1.0
    TIME_STEP = 1.0 / 60.0
    THRUST_POWER = 40.0     # Max force from each thruster
    FUEL_COST = 0.01        # Fuel cost per Newton of thrust
    INITIAL_FUEL = 100.0    # Starting fuel

    # Lander dimensions
    LANDER_WIDTH = 3.0
    LANDER_HEIGHT = 1.0
    LANDER_MASS = 10.0
    LANDER_COF = 1.5

    # Sensor specifications
    LASER_RANGE = 100.0

    # Rendering
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    REPLAY_SPEED = 1.0
    RENDER_SCALE = 10.0

    # RL
    MAX_EPISODE_DURATION = 20.0

    # Calculated
    PHYSICS_STEPS_PER_FRAME = int(1.0 / TIME_STEP / FPS)
    RENDER_TIME_STEP = 1 / FPS
