#!/usr/bin/env python3
"""
multi_visualization.py

This script loads a checkpoint (from a model file) and instantiates multiple agent-environment pairs.
One agent runs in deterministic mode (drawn in blue) while several copies run stochastically (illustrated
in green semi-transparently). Their states are simulated concurrently and drawn over the same GUI window.
The target seeds are set so that each concurrently simulated actor has the same objective.
Optionally, the rendered frames can be saved to disk.
"""

import logging
import os
import sys
import argparse
import importlib
import pygame
import numpy as np
from datetime import datetime

from lander_learner.environment import LunarLanderEnv
from lander_learner.gui import LunarLanderGUI
from lander_learner import scenarios
from lander_learner.utils.helpers import load_scenarios

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s", stream=sys.stdout)


# ---------------------------
# Main simulation entry point
# ---------------------------
def main():
    # Load scenario defaults.
    try:
        with importlib.resources.path(scenarios, "scenarios.json") as scn_path:
            scenario_list = load_scenarios(scn_path)
    except RuntimeError:
        logger.fatal("Error loading scenarios", exc_info=True)
        sys.exit(1)

    # Parse all arguments, using the appropriate scenario to set defaults.
    try:
        args = parse_args()
        scenario = scenario_list[args.scenario]
        del scenario_list
        reward_function = scenario.get("reward_function", "default")
        observation_function = scenario.get("observation_function", "default")
        target_zone = scenario.get("target_zone", False)
    except ValueError:
        logger.fatal("Error parsing arguments", exc_info=True)
        sys.exit(1)
    except KeyError:
        logger.fatal(f"Error: scenario '{args.scenario}' not found in scenarios.json", exc_info=True)
        sys.exit(1)

    # Dynamically import the agent module.
    agent_module = importlib.import_module(f"lander_learner.agents.{args.agent_type.lower()}_agent")
    AgentClass = getattr(agent_module, f"{args.agent_type.upper()}Agent")

    env_agents = []
    env_seed = int(np.random.SeedSequence().generate_state(1)[0])

    # Create stochastic agent–env pairs.
    for _ in range(args.num_stochastic):
        env_stoch = LunarLanderEnv(
            gui_enabled=True,
            reward_function=reward_function,
            observation_function=observation_function,
            target_zone=target_zone,
            seed=env_seed
        )
        agent_stoch = AgentClass(env_stoch, deterministic=False)
        agent_stoch.load_model(args.checkpoint)
        # Use a different colour (e.g. green) and semi-transparency.
        style_stoch = {"color": (0, 255, 0), "alpha": 128}
        env_agents.append((env_stoch, agent_stoch, style_stoch))

    # Create the deterministic agent–env pair.
    env_det = LunarLanderEnv(
        gui_enabled=True,
        reward_function=reward_function,
        observation_function=observation_function,
        target_zone=target_zone,
        seed=env_seed
    )
    agent_det = AgentClass(env_det, deterministic=True)
    agent_det.load_model(args.checkpoint)
    # Use a distinct colour (e.g. blue) and full opacity.
    style_det = {"color": (0, 0, 255), "alpha": 255}
    env_agents.append((env_det, agent_det, style_det))

    # Initialize our multi-agent GUI.
    gui = LunarLanderGUI(
        [env for env, _, _ in env_agents],
        multi_mode=True,
        styles=[style for _, _, style in env_agents]
    )

    # Optionally, set up recording.
    if args.record:
        record_dir = "data/recordings/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(record_dir, exist_ok=True)
        frame_count = 0

    # Main simulation loop.
    running = [True] * len(env_agents)
    episode_count = 0
    while True:
        # For each agent/environment pair, step the simulation.
        for i, (env, agent, _) in enumerate(env_agents):
            # (We assume each environment is in inference mode.)
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
        gui.render()

        # If recording, save a screenshot.
        if args.record:
            frame_path = os.path.join(record_dir, f"frame_{frame_count:06d}.png")
            pygame.image.save(gui.screen, frame_path)
            frame_count += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize multiple agent runs from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.zip file or directory with checkpoint zip file).")
    parser.add_argument("--agent_type", type=str, default="PPO",
                        help="Agent type to use (e.g., PPO or SAC).")
    parser.add_argument("--num_stochastic", type=int, default=3,
                        help="Number of stochastic agent copies to run.")
    parser.add_argument("--record", action="store_true",
                        help="Save rendered frames to disk (in a subfolder 'recordings').")
    parser.add_argument("--scenario", type=str, default="default",
                        help="Scenario to use (e.g., 'default' or 'landing').")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
