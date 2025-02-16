class BaseAgent:
    """
    Base class for RL agents.
    Subclasses should override the train, get_action, save_model, and load_model methods.
    """

    def __init__(self, env, deterministic=True):
        """
        env: The Gym environment.
        deterministic: Whether the agent should act deterministically.
        """
        self.env = env
        self.deterministic = deterministic
        self.model = None

    def train(self, timesteps):
        """
        Train the agent for a specified number of timesteps.
        """
        raise NotImplementedError("Subclasses must implement train()")

    def get_action(self, observation):
        """
        Given an observation, return an action.
        """
        raise NotImplementedError("Subclasses must implement get_action()")

    def save_model(self, path):
        """
        Save the model to the specified path.
        """
        raise NotImplementedError("Subclasses must implement save_model()")

    def load_model(self, path):
        """
        Load the model from the specified path.
        """
        raise NotImplementedError("Subclasses must implement load_model()")
