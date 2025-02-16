import sys
import pytest
from lander_learner.utils.parse_args import parse_args

def test_parse_args_defaults(monkeypatch):
    # Provide a dummy scenarios dictionary.
    scenarios = {
        "base": {
            "rl_agent_type": "PPO",
            "reward_function": "rightward",
            "observation_function": "default",
            "target_zone": False,
            "learning_frames": 10000
        }
    }
    test_args = ["program", "--scenario", "base"]
    monkeypatch.setattr("sys.argv", test_args)
    args = parse_args(scenarios)
    # Check that the defaults are correctly set from the scenario.
    assert args.rl_agent.upper() == "PPO"
    assert args.reward_function == "rightward"
    assert args.observation_function == "default"
