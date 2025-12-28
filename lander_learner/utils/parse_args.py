import argparse
from lander_learner.utils.config import RL_Config


def parse_args(scenarios: dict) -> argparse.Namespace:
    """Parses command-line arguments using scenario defaults.

    This function performs a preliminary parse to extract the scenario name from the
    command line so that default parameters for that scenario (loaded from a JSON file)
    can be applied to subsequent argument definitions. It then builds a full parser with
    arguments for GUI mode, training mode, agent type, reward and observation functions,
    target zone settings, and model paths.

    Args:
        scenarios (dict): A dictionary of scenario configurations, typically loaded from
            "scenarios.json". Each key is a scenario name and each value is a dictionary of
            default parameters such as "rl_agent_type", "reward_function", "observation_function",
            "target_zone", and "learning_frames".

    Returns:
        argparse.Namespace: An argparse namespace containing all parsed command-line arguments.

    Raises:
        ValueError: If the scenario specified in the command line is not found in the scenarios dictionary.
    """
    # Preliminary parser to extract the scenario argument.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--scenario",
        type=str,
        default="base",
        help="Scenario name to use (as defined in scenarios.json)"
    )
    # Only the scenario argument is parsed here.
    args, _ = parser.parse_known_args()

    scenario_name = args.scenario
    if scenario_name not in scenarios:
        raise ValueError(f"Scenario '{scenario_name}' not found in scenario file (scenarios/scenarios.json).")

    # Extract default values from the chosen scenario.
    scenario_defaults = scenarios[scenario_name]
    default_agent_type = scenario_defaults.get("agent_type", "PPO")
    default_reward_function = scenario_defaults.get("reward_function", "default")
    default_observation_function = scenario_defaults.get("observation_function", "default")
    default_level_name = scenario_defaults.get("level_name", "half_plane")
    default_target_zone = scenario_defaults.get("target_zone", None)
    default_learning_frames = scenario_defaults.get("learning_frames", 10000)

    # Build the full argument parser with scenario-based defaults.
    parser = argparse.ArgumentParser(
        description="2D Lunar Lander with Scenario Selection and Conditional Imports"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable GUI Rendering (only available in single environment mode)"
    )
    parser.add_argument(
        "--mode",
        choices=["human", "train", "inference"],
        default="human",
        help="Mode to run the environment in (human|train|inference)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (for human/inference modes)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=default_learning_frames,
        help="Number of training timesteps (only for train mode)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of environments to run in parallel during training"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="base_scenario",
        help="Scenario name to use (as defined in scenarios.json)"
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default=default_agent_type,
        help="RL agent type to use (overrides scenario default)"
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default=default_reward_function,
        help="Reward function to use (overrides scenario default)"
    )
    parser.add_argument(
        "--observation_function",
        type=str,
        default=default_observation_function,
        help="Observation function to use (overrides scenario default)"
    )
    parser.add_argument(
        "--level_name",
        type=str,
        default=default_level_name,
        help="Level preset name to use (overrides scenario default)"
    )
    parser.add_argument(
        "--target_zone",
        action="store_false" if default_target_zone else "store_true",
        help="Enable target zone mode (overrides scenario default)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=RL_Config.DEFAULT_CHECKPOINT_DIR,
        help="Path to save/load the model (for train and inference mode)"
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to load a model checkpoint from (for continued training)"
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Run inference in multi-render mode (multiple agent-environment pairs concurrently)"
    )
    parser.add_argument(
        "--num_stochastic",
        type=int,
        default=3,
        help="Number of stochastic agent copies to run in multi-render mode"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save rendered frames to disk (in a subfolder 'recordings/date-time-now')"
    )
    return parser.parse_args()
