import numpy as np
import pygame

class HumanAgent:
    """
    Provides a simple interface to read user input for thruster control.
    Integrated with the pygame event loop in `gui.py`.
    """

    def __init__(self, env):
        self.env = env
        self.state_flags = {
            "left_thruster": False,
            "right_thruster": False
        }

    def get_action(self, observation):
        """
        Return a 2D action vector: [left_thruster, right_thruster].
        Use keyboard input state flags to toggle thrust on and off.
        """
        return np.array([
            -1.0 + 2.0 * self.state_flags["left_thruster"], 
            -1.0 + 2.0 * self.state_flags["right_thruster"]
            ], dtype=np.float32)
    
    def handle_key_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.state_flags["left_thruster"] = True
            elif event.key == pygame.K_RIGHT:
                self.state_flags["right_thruster"] = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                self.state_flags["left_thruster"] = False
            elif event.key == pygame.K_RIGHT:
                self.state_flags["right_thruster"] = False
