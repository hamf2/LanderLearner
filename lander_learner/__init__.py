"""
LanderLearner Package

This package provides a complete framework for simulating, training, and evaluating
reinforcement learning agents on a 2D lunar lander task. It encompasses several key modules:

  - environment: Implements the LunarLanderEnv, a Gym-compatible environment for simulating
    the lander's physics and dynamics.
  - physics: Contains the physics engine built with pymunk for simulating realistic motion and collisions.
  - rewards: Defines various reward functions and utilities for shaping agent behavior.
  - observations: Provides observation generators and wrappers for constructing state representations
    from the environment.
  - agents: Implements different reinforcement learning agents (e.g., PPO, SAC) and a human-controlled agent.
  - utils: Offers utility functions, configuration management, and helper methods for tasks such as
    argument parsing and checkpoint management.

Use this package to experiment with and develop RL algorithms for lunar landing tasks. The modular
design allows for easy customization of reward schemes, observation spaces, and agent behaviors.
"""
