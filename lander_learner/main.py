#!/usr/bin/env python3
import sys
import os
import importlib
import logging

from lander_learner import scenarios
from lander_learner.environment import LunarLanderEnv
from lander_learner.utils.rl_config import RL_Config
from lander_learner.utils.helpers import load_scenarios
from lander_learner.utils.parse_args import parse_args

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(created)f: %(message)s", stream=sys.stdout)


# RL agent and GUI modules are imported conditionally based on the mode.


def main():
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
    except ValueError:
        logger.fatal("Error parsing arguments", exc_info=True)
        sys.exit(1)

    # --- Conditional Imports ---
    # Import RL agent modules only if needed.
    if args.mode in ["train", "inference"]:
        RL_AGENT_MAP = {
            "PPO": [
                importlib.import_module("lander_learner.agents.ppo_agent").PPOAgent,
                {"device": RL_Config.PPO_DEVICE},
            ],
            "SAC": [
                importlib.import_module("lander_learner.agents.sac_agent").SACAgent,
                {"device": RL_Config.SAC_DEVICE},
            ],
            # Additional agents here.
        }
        args.rl_agent = args.rl_agent.upper()
        if args.rl_agent not in RL_AGENT_MAP:
            logger.warning(f"Warning: RL agent '{args.rl_agent}' not found. Defaulting to PPO.")
            args.rl_agent = "PPO"
        agent_class, agent_options = RL_AGENT_MAP.get(args.rl_agent, RL_AGENT_MAP["PPO"])
    # For training mode, import the vectorized environment.
    if args.mode == "train":
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
        )  # Alternative: SubprocVecEnv - DummyVecEnv is faster in this case.
    # For human mode, import the human agent.
    if args.mode == "human":
        HumanAgent = importlib.import_module("lander_learner.agents.human_agent").HumanAgent

    # Import GUI only if needed.
    if args.gui:
        LunarLanderGUI = importlib.import_module("lander_learner.gui").LunarLanderGUI

    # --- Environment and Agent Setup ---
    if args.mode == "train":
        # Training mode: use a vectorized environment.
        env = DummyVecEnv(
            [
                lambda: LunarLanderEnv(
                    gui_enabled=False,
                    reward_function=args.reward_function,
                    observation_function=args.observation_function,
                    target_zone=args.target_zone,
                )
                for _ in range(args.num_envs)
            ]
        )
        agent = agent_class(env, **agent_options)
        if args.load_checkpoint:
            agent.load_model(args.load_checkpoint)
        try:
            agent.train(args.timesteps)
        except KeyboardInterrupt:
            logger.warning("Training interrupted.")
        agent.save_model(args.model_path)
        env.close()
        sys.exit(0)
    else:
        # For human or inference mode, use a single environment instance.
        env = LunarLanderEnv(
            gui_enabled=args.gui,
            reward_function=args.reward_function,
            observation_function=args.observation_function,
            target_zone=args.target_zone,
        )
        if args.mode == "human":
            agent = HumanAgent(env)
        else:  # inference mode
            agent = agent_class(env, **agent_options)
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

        logger.info(f"Episode {episode + 1} finished with total reward: {total_reward}")
        agent.deterministic = False  # Ensure agent is stochastic after first episode.

    env.close()


if __name__ == "__main__":
    main()
