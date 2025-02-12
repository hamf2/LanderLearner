#!/usr/bin/env python3
import sys
import importlib
from environment import LunarLanderEnv
from utils.config import Config
from utils.helpers import load_scenarios
from utils.parse_args import parse_args
# RL agent and GUI modules are imported conditionally based on the mode.

def main():

    # --- Scenario and Argument Parsing ---
    # Load scenario defaults.
    try:
        scenarios = load_scenarios(Config.SCENARIO_FILE)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    # Parse all arguments, using the appropriate scenario to set defaults.
    try:
        args = parse_args(scenarios)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # --- Conditional Imports ---
    # Import RL agent modules only if needed.
    if args.mode in ["train", "inference"]:
        RL_AGENT_MAP = {
            "PPO": importlib.import_module("agents.rl_agent").RLAgent,
            # Additional agents here.
        }
        if args.rl_agent not in RL_AGENT_MAP:
            print(f"Warning: RL agent '{args.rl_agent}' not found. Defaulting to PPO.", file=sys.stderr)
        agent_class = RL_AGENT_MAP.get(args.rl_agent, RL_AGENT_MAP["PPO"])
    # For training mode, import the vectorized environment.
    if args.mode == "train":
        from stable_baselines3.common.vec_env import DummyVecEnv # Alternative: SubprocVecEnv - DummyVecEnv is faster in this case.
    # For human mode, import the human agent.
    if args.mode == "human":
        HumanAgent = importlib.import_module("agents.human_agent").HumanAgent

    # Import GUI only if needed.
    if args.gui:
        LunarLanderGUI = importlib.import_module("gui").LunarLanderGUI

    # --- Environment and Agent Setup ---
    if args.mode == "train":
        # Training mode: use a vectorized environment.
        env = DummyVecEnv([
            lambda: LunarLanderEnv(gui_enabled=False, reward_function=args.reward_function, observation_function=args.observation_function, target_zone=args.target_zone)
            for _ in range(args.num_envs)
        ])
        agent = agent_class(env)
        agent.train(args.timesteps)
        agent.save_model(args.model_path)
        env.close()
        sys.exit(0)
    else:
        # For human or inference mode, use a single environment instance.
        env = LunarLanderEnv(gui_enabled=args.gui, reward_function=args.reward_function, observation_function=args.observation_function, target_zone=args.target_zone)
        if args.mode == "human":
            agent = HumanAgent(env)
        else:  # inference mode
            agent = RL_AGENT_MAP.get(args.rl_agent, RL_AGENT_MAP["PPO"])(env)
            agent.load_model(args.model_path)

        if args.gui:
            gui = LunarLanderGUI(env)
            if args.mode == "human" and hasattr(agent, "handle_key_event"):
                gui.set_key_callback(agent.handle_key_event)

    # --- Run Episodes ---
    # Run the specified number of episodes for human or inference mode.
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if args.gui:
                gui.render()

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()