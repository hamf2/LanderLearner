# 2D Lunar Lander

A modular Python project using:
- **pymunk** for 2D physics (Chipmunk2D bindings)
- **pygame** for optional GUI rendering
- **gymnasium** for environment creation
- **stable-baselines3** for RL training
- **numpy** for numerical computations

## Directory Structure
lunar_lander/ 

├── main.py 

├── environment.py 

├── gui.py 

├── physics.py 

├── agents/ 

│ ├── human_agent.py 

│ ├── rl_agent.py 

├── utils/ 

│ ├── config.py 

│ ├── helpers.py 

│ ├── parse_args.py

│ ├── rewards.py

├── models/ 

├── scenarios/

│ ├── scenarios.json

└── README.md

## Installation
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/macOS
   # or:
   venv\Scripts\activate      # On Windows
2. Install package:
   pip install -r requirements.txt
   pip install -e .
3. Run the main script:
   For a human-controlled GUI:
   `lander_learner --gui --mode human`
   
   For RL training:
   `lander_learner --mode train`
   
   For RL inference (after a model is saved):
   `lander_learner --gui --mode inference`

   For TensorBoard log viewing:
   `tensorboard --logdir=data/logs/lander_tensorboard`

   For multiple agent visualisation (renders multiple stachastic instances simultaneously):
   `python lander_learner/utils/multiple_render.py --checkpoint [path] --agent_type [agent_type]`


# Expanding This Project (possible directions)

Physics: Adjust or refine the shapes, collision detection, custom terrain.
Rendering: Draw terrain accurately, display thruster flames, etc.
Reward Shaping: Add partial rewards for stable flight, gentle landings, etc.
Action Space: Use discrete or continuous thrusters, possibly rotational thrusters.
Multiple Agents: Investigate multi-agent RL or competing landers.

---

# Final Notes

1. **Modularity**  
   - `environment.py` is **gym-compatible** for training or manual/human mode.  
   - `gui.py` is **decoupled** from the environment, only reading its state.  
   - `physics.py` handles **all** physics updates with pymunk.  
   - `agents/` can be expanded with different agents (human, RL, scripted).  
   - `utils/` keeps config constants and helper functions. 
   - `utils/rewards.py` enables interchangable reward functions.
   - `utils/observations.py` enables interchangable observation modes.
   - `utils/target.py` implements multiple options for target spawning and motion. 

2. **Optional Headless Mode**  
   - Passing `--gui` toggles the UI. Without it, training can proceed faster.  