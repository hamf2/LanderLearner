#!/usr/bin/env python3
"""
main.py

This is the main entry point for LanderLearner, a lunar lander simulation and training framework.
It parses command-line arguments, loads scenario defaults, conditionally imports RL agents and GUI
modules, sets up the environment, and runs episodes in training, inference, or human-interactive mode.
"""
import sys
import os
import importlib
import logging
import numpy as np

from lander_learner import scenarios
from lander_learner.environment import LunarLanderEnv
from lander_learner.utils.config import RL_Config
from lander_learner.utils.helpers import load_scenarios
from lander_learner.utils.parse_args import parse_args  # see updated parse_args below

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s", stream=sys.stdout)


# ---------------------------
# Main simulation entry point
# ---------------------------
def main():
    """Main entry point for LanderLearner.

    The function performs the following steps:
      1. Logs the working directory.
      2. Loads scenario defaults from a JSON file.
      3. Parses command-line arguments using the scenario defaults.
      4. Conditionally imports RL agent modules, vectorized environments, and GUI modules based on the mode.
      5. Sets up the environment and agent based on the chosen mode (train, human, or inference).
      6. Runs episodes, executing agent actions and rendering (if enabled).
      7. Logs episode rewards and ensures proper cleanup.

    Raises:
        SystemExit: If scenarios cannot be loaded or arguments are parsed incorrectly.
    """
    logger.info("LanderLearner started in %s", os.getcwd())

    # --- Scenario and Argument Parsing ---
    # Load scenario defaults.
    try:
        with importlib.resources.path(scenarios, "scenarios.json") as scn_path:
            scenario_list = load_scenarios(scn_path)
    except RuntimeError:
        logger.fatal("Error loading scenarios", exc_info=True)
        sys.exit(1)

    # Parse all arguments, using the appropriate scenario to set defaults.
    try:
        args = parse_args(scenario_list)
        del scenario_list
    except (ValueError, KeyError):
        logger.fatal("Error parsing arguments", exc_info=True)
        sys.exit(1)

    # --- Conditional Imports ---
    if args.mode in ["train", "inference"]:
        # Dynamically import the agent module.
        try:
            agent_module = importlib.import_module(f"lander_learner.agents.{args.agent_type.lower()}_agent")
            AgentClass = getattr(agent_module, f"{args.agent_type.upper()}Agent")
            agent_options = getattr(RL_Config, f"{args.agent_type.upper()}_OPTIONS")
        except (ImportError, AttributeError):
            logger.fatal(f"Error importing agent module or class for agent type: {args.agent_type}", exc_info=True)
            sys.exit(1)
    # For training mode, import vectorized environment.
    if args.mode == "train":
        from stable_baselines3.common.vec_env import DummyVecEnv
    # For human mode, import the human agent.
    if args.mode == "human":
        HumanAgent = importlib.import_module("lander_learner.agents.human_agent").HumanAgent
    # Import GUI only if needed.
    if args.gui:
        LunarLanderGUI = importlib.import_module("lander_learner.gui").LunarLanderGUI

    # --- Environment and Agent Setup ---
    if args.mode == "train":
        logging.info("Running in training mode")
        # Training mode: use a vectorized environment.
        env = DummyVecEnv([
            lambda: LunarLanderEnv(
                gui_enabled=False,
                reward_function=args.reward_function,
                observation_function=args.observation_function,
                target_zone=args.target_zone,
            )
            for _ in range(args.num_envs)
        ])
        agent = AgentClass(env, **agent_options)
        if args.load_checkpoint:
            agent.load_model(args.load_checkpoint)
        try:
            agent.train(args.timesteps, checkpoint_freq=RL_Config.CHECKPOINT_FREQ // args.num_envs)
        except KeyboardInterrupt:
            logger.warning("Training interrupted.")
        agent.save_model(args.model_path)
        env.close()
        sys.exit(0)
    if args.mode == "human" or (args.mode == "inference" and not args.multi):
        logger.info("Running in %s mode", args.mode)
        # Use single environment mode.
        env = LunarLanderEnv(
            gui_enabled=args.gui,
            reward_function=args.reward_function,
            observation_function=args.observation_function,
            target_zone=args.target_zone,
        )
        if args.mode == "human":
            agent = HumanAgent(env)
        else:
            agent = AgentClass(env, **agent_options)
            agent.load_model(args.model_path)
        if args.gui:
            gui = LunarLanderGUI(env)
            # Set key callback for human mode if applicable.
            if hasattr(agent, "handle_key_event"):
                gui.set_key_callback(agent.handle_key_event)

        # --- Run Episodes ---
        # Execute the specified number of episodes for human or inference mode.
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

            logger.info(f"Episode {episode + 1} finished with total reward: {total_reward}")
            # Ensure agent becomes stochastic after the first episode.
            agent.deterministic = False

        env.close()
    elif args.multi:
        logger.info("Running in multi-render mode")
        # --- Multi-render mode ---
        env_agents = []
        # Use a common seed for all environments.
        env_seed = int(np.random.SeedSequence().generate_state(1)[0])
        # Create stochastic agent–env pairs.
        for _ in range(args.num_stochastic):
            env_stoch = LunarLanderEnv(
                gui_enabled=True,
                reward_function=args.reward_function,
                observation_function=args.observation_function,
                target_zone=args.target_zone,
                seed=env_seed
            )
            agent_stoch = AgentClass(env_stoch, deterministic=False, **agent_options)
            agent_stoch.load_model(args.model_path)
            # Use green with semi-transparency.
            style_stoch = {"color": (0, 255, 0), "alpha": 128}
            env_agents.append((env_stoch, agent_stoch, style_stoch))
        # Create deterministic agent–env pair.
        env_det = LunarLanderEnv(
            gui_enabled=True,
            reward_function=args.reward_function,
            observation_function=args.observation_function,
            target_zone=args.target_zone,
            seed=env_seed
        )
        agent_det = AgentClass(env_det, deterministic=True, **agent_options)
        agent_det.load_model(args.model_path)
        # Use blue with full opacity.
        style_det = {"color": (0, 0, 255), "alpha": 255}
        env_agents.append((env_det, agent_det, style_det))
        # Initialize multi-agent GUI.
        if args.gui:
            gui = LunarLanderGUI(
                [env for env, _, _ in env_agents],
                multi_mode=True,
                styles=[style for _, _, style in env_agents]
            )
        # Simulation loop.
        running = [True] * len(env_agents)
        episode_count = 0
        while True:
            for i, (env, agent, _) in enumerate(env_agents):
                if running[i]:
                    obs = env._get_observation()  # using the internal observation getter
                    action = agent.get_action(obs)
                    obs, reward, done, truncated, info = env.step(action)
                if done:
                    running[i] = False
            if not any(running):
                episode_count += 1
                logger.info(f"Episode {episode_count} finished.")
                if episode_count >= args.episodes:
                    break
                running = [True] * len(env_agents)
                env_seed = int(np.random.SeedSequence().generate_state(1)[0])
                for env, _, _ in env_agents:
                    env.reset(seed=env_seed)
            if args.gui:
                gui.render()
        for env, _, _ in env_agents:
            env.close()


if __name__ == "__main__":
    main()
