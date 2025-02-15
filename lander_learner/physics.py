import pymunk
import numpy as np
from lander_learner.utils.config import Config


class PhysicsEngine:
    """
    Manages Chipmunk2D/pymunk-based simulation.
    """

    def __init__(self):
        # Create the pymunk Space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -Config.GRAVITY)

        self._create_ground()
        self._create_lander()

        # Add collision handler callback
        collision_handler = self.space.add_default_collision_handler()
        collision_handler.begin = self._collision_begin_callback
        collision_handler.separate = self._collision_separate_callback
        collision_handler.post_solve = self._collision_post_solve_callback
        self.collision_state = False
        self.collision_impulse = 0.0

    def reset(self):
        """
        Reset the physics state for a new episode.
        Bodies removed and re-added as needed for clean state.
        Variables are reset as needed.
        """
        self.space.remove(self.lander_body, self.lander_shape)
        self._create_lander()
        self._move_ground()

        self.collision_state = False
        self.collision_impulse = 0.0

    def update(self, left_thruster, right_thruster, env):
        """
        Advance the physics simulation by one time step,
        applying thruster forces to the lander.
        """

        if env.fuel_remaining > 0.0:
            # Convert thruster power to force
            thruster_force_left = (left_thruster + 1.0)/2.0 * Config.THRUST_POWER
            thruster_force_right = (right_thruster + 1.0)/2.0 * Config.THRUST_POWER

            # Apply upward forces on opposite corners of the lander
            # Here force applied in body coordinates.
            self.lander_body.apply_force_at_local_point(
                (0, thruster_force_left), (-Config.LANDER_WIDTH/2, 0)
            )
            self.lander_body.apply_force_at_local_point(
                (0, thruster_force_right), (Config.LANDER_WIDTH/2, 0)
            )

            # Decrease fuel in env according to consumption rate
            fuel_used = (thruster_force_left + thruster_force_right) * Config.FUEL_COST
            env.fuel_remaining = max(0.0, env.fuel_remaining - fuel_used)

        if (
            self.lander_body.position.x - (self.ground_body.position.x + self.ground_shape.a.x) < Config.LANDER_WIDTH 
            or self.lander_body.position.x - (self.ground_body.position.x + self.ground_shape.b.x) > - Config.LANDER_WIDTH
            ):

            self._move_ground()

        # Step the physics simulation
        for _ in range(Config.PHYSICS_STEPS_PER_FRAME):
            self.space.step(Config.TIME_STEP)

        # Update env state from pymunk body
        env.lander_position = np.array(self.lander_body.position, dtype=np.float32)
        env.lander_velocity = np.array(self.lander_body.velocity, dtype=np.float32)
        env.lander_angle = np.array(self.lander_body.angle, dtype=np.float32)
        env.lander_angular_velocity = np.array(self.lander_body.angular_velocity, dtype=np.float32)
        env.collision_state = self.collision_state
        env.collision_impulse = self.collision_impulse
        
    def _create_ground(self):
        """
        Create a static body for the ground.
        TODO: Replace with more complex terrain / random variation.
        """
        self.ground_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.ground_body.position = (0, 0)
        self.ground_shape = pymunk.Segment(self.ground_body, (-100, 0), (100, 0), 0.1)
        self.ground_shape.friction = 1.0
        self.ground_shape.elasticity = 0.5
        self.space.add(self.ground_body, self.ground_shape)

    def _move_ground(self):
        """
        Move the ground to a new position centred under the lander.
        """
        self.space.remove(self.ground_body, self.ground_shape)
        self.ground_body.position = (self.lander_body.position.x, 0)
        self.space.add(self.ground_body, self.ground_shape)
    
    def _create_lander(self):
        """
        Create a dynamic body for the lander.
        """
        lander_moment = pymunk.moment_for_box(Config.LANDER_MASS, (Config.LANDER_WIDTH, Config.LANDER_HEIGHT))
        self.lander_body = pymunk.Body(Config.LANDER_MASS, lander_moment, body_type=pymunk.Body.DYNAMIC)
        self.lander_body.sleep_threshold = 0.1
        self.lander_body.position = (0, 10)
        self.lander_shape = pymunk.Poly.create_box(self.lander_body, (Config.LANDER_WIDTH, Config.LANDER_HEIGHT), radius=0.1)
        self.lander_shape.friction = Config.LANDER_COF
        self.lander_shape.elasticity = 0.5
        self.space.add(self.lander_body, self.lander_shape)

    def _collision_begin_callback(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict):
        """
        Called when collisions begin. Sets the collision state.
        """
        self.collision_state = True
        return True
    
    def _collision_separate_callback(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict):
        """
        Called when collisions end. Resets the collision state.
        """
        self.collision_state = False
        return True
    
    def _collision_post_solve_callback(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict):
        """
        Called after collision resolution. Records the impulse.
        """
        self.collision_impulse = max(self.collision_impulse, arbiter.total_impulse.length)
        return True