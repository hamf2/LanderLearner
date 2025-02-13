import argparse
from utils.config import Config

def parse_args(scenarios: dict) -> argparse.Namespace:
    """
    Perform a minimal (preliminary) parse to extract the scenario name.
    This allows us to load the corresponding defaults from scenarios.json.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scenario", type=str, default="base",
                        help="Scenario name to use (as defined in scenarios.json)")
    # We only care about the scenario argument here.
    args, _ = parser.parse_known_args()

    scenario_name = args.scenario

    if scenario_name not in scenarios:
        raise ValueError(f"Scenario '{scenario_name}' not found in {Config.SCENARIO_FILE}.")
    
    # Extract defaults from the chosen scenario.
    scenario_defaults = scenarios[scenario_name]

    default_rl_agent = scenario_defaults.get("rl_agent_type", "PPO")
    default_reward_function = scenario_defaults.get("reward_function", "default")
    default_observation_function = scenario_defaults.get("observation_function", "default")
    default_target_zone = scenario_defaults.get("target_zone", None)
    default_learning_frames = scenario_defaults.get("learning_frames", 10000)

    parser = argparse.ArgumentParser(
        description="2D Lunar Lander with Scenario Selection and Conditional Imports"
    )
    parser.add_argument("--gui", action="store_true",
                        help="Enable GUI Rendering (only available in single environment mode)")
    parser.add_argument("--mode", choices=["human", "train", "inference"],
                        default="human",
                        help="Mode to run the environment in (human|train|inference)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run (for human/inference modes)")
    parser.add_argument("--timesteps", type=int, default=default_learning_frames,
                        help="Number of training timesteps (only for train mode)")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Number of environments to run in parallel during training")
    parser.add_argument("--scenario", type=str, default="base_scenario",
                        help="Scenario name to use (as defined in scenarios.json)")
    parser.add_argument("--rl_agent", type=str, default=default_rl_agent,
                        help="RL agent type to use (overrides scenario default)")
    parser.add_argument("--reward_function", type=str, default=default_reward_function,
                        help="Reward function to use (overrides scenario default)")
    parser.add_argument("--observation_function", type=str, default=default_observation_function,
                        help="Observation function to use (overrides scenario default)")
    parser.add_argument("--target_zone", action="store_false" if default_target_zone else "store_true",
                        help="Enable target zone mode (overrides scenario default)")
    parser.add_argument("--model_path", type=str, default=Config.DEFAULT_CHECKPOINT_DIR,
                        help="Path to save/load the model (for train and inference mode)")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to load a model checkpoint from (for continued training)")
    return parser.parse_args()
    