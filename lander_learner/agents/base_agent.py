class BaseAgent:
    """Base class for RL agents.

    This class defines the common interface for RL agents interacting with the Lunar Lander environment.
    Subclasses should override the methods:
      - train
      - get_action
      - save_model
      - load_model

    Attributes:
        env: The Gym environment the agent interacts with.
        deterministic (bool): If True, the agent acts deterministically.
        model: The underlying model for the agent (to be set by subclasses).
    """

    def __init__(self, env, deterministic=True):
        """Initializes a BaseAgent instance.

        Args:
            env: The Gym environment instance.
            deterministic (bool, optional): Flag to specify whether the agent acts deterministically.
                Defaults to True.
        """
        self.env = env
        self.deterministic = deterministic
        self.model = None

    def train(self, timesteps):
        """Trains the agent for a specified number of timesteps.

        Args:
            timesteps (int): The number of timesteps to train for.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement train()")

    def get_action(self, observation):
        """Computes and returns an action given an observation.

        Args:
            observation: The observation from the environment.

        Returns:
            The action computed by the agent.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement get_action()")

    def save_model(self, path):
        """Saves the agent's model to the specified path.

        Args:
            path (str): The file path to save the model.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement save_model()")

    def load_model(self, path):
        """Loads the agent's model from the specified path.

        Args:
            path (str): The file path to load the model from.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement load_model()")
