import numpy as np
import pygame
from lander_learner.agents.base_agent import BaseAgent


class HumanAgent(BaseAgent):
    """An RL agent that obtains actions from human input.

    This agent reads user input via keyboard events to control the lander thrusters.
    It is integrated with the pygame event loop in the GUI.
    """

    def __init__(self, env):
        """Initializes a HumanAgent instance.

        Args:
            env: The Gym environment instance.
        """
        super().__init__(env, deterministic=True)
        self.state_flags = {"left_thruster": False, "right_thruster": False}

    def get_action(self, observation):
        """Returns an action based on the current key state flags.

        The action is a 2D vector computed from the state flags for the left and right thrusters.

        Args:
            observation: The current observation (unused in this agent).

        Returns:
            numpy.ndarray: A 2-element action vector.
        """
        return np.array(
            [
                -1.0 + 2.0 * self.state_flags["left_thruster"],
                -1.0 + 2.0 * self.state_flags["right_thruster"]
            ],
            dtype=np.float32,
        )

    def handle_key_event(self, event: pygame.event.Event):
        """Handles keyboard events to update thruster state flags.

        Args:
            event (pygame.event.Event): The keyboard event.
        """
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

    def train(self, timesteps):
        """Training method for HumanAgent.

        Human agents do not train automatically; this method is a no-op.

        Args:
            timesteps (int): The number of timesteps (unused).
        """
        pass

    def save_model(self, path):
        """No-op for saving model in HumanAgent.

        Args:
            path (str): The path where a model would be saved (unused).
        """
        pass

    def load_model(self, path):
        """No-op for loading model in HumanAgent.

        Args:
            path (str): The path from which a model would be loaded (unused).
        """
        pass
