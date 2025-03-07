# Package Demonstration Readme

Welcome to the package demonstration. This document gives a brief overview of the package features and intro to the reinforcement learning agents used.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Features](#features)
- [Performance Videos](#performance-videos)
- [Algorithm Explanations in Brief](#algorithm-explanations-in-brief)
    - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
    - [Soft Actor-Critic (SAC)](#soft-actor-critic-sac)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Introduction

This package is designed to bridge the gap between human intuition and model performance using cutting-edge deep reinforcement learning algorithms. The modular design ensures ease of integration and extensive customization options.

## Installation

Install the package via the following commands:

```bash
git clone https://github.com/hamf2/LanderLearner.git
cd LanderLearner
pip install .
```

For more detailed installation instructions, please refer to the [Installation Guide](./INSTALL.md).

## Features

- Modular architecture for reinforcement learning algorithms.
- Integrated performance evaluation with video recording.
- Extensible plugin support for custom metrics and reward functions.

## Performance Videos

### Human Performance Videos

Embed or link videos showcasing human performance benchmarks:

- [Human Performance Video 1](#)

### Model Performance Videos

Embed or link videos demonstrating model performance:

- [Model Performance Video 1](#)

## Algorithm Explanations in Brief

### Proximal Policy Optimization (PPO)

PPO optimizes a "surrogate" objective function to limit policy updates at each iteration preventing large changes:

The **clipped surrogate objective** is given by:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \Big[ \min\Big( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \Big) \Big],
$$

where

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

and $ \hat{A}_t $ is an estimator of the advantage function, $ \epsilon $ is a hyperparameter that controls the update step, typically set to a small value such as 0.1 or 0.2.

PPO’s strategy lies in balancing exploration and exploitation while ensuring stability in the policy updates.

### Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic algorithm that incorporates a maximum entropy objective to promote exploration by encouraging policies to act as randomly as possible while still maximizing reward.

The objective for the policy is:

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left[ r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right],
$$

where:
- $ \mathcal{H}(\pi(\cdot|s_t)) $ is the entropy of the policy at state $ s_t $.
- $ \alpha $ is the temperature parameter that balances the trade-off between maximizing the reward and maximizing entropy.
- $ \rho_\pi $ denotes the state-action marginal distribution under policy $ \pi $.

SAC leverages two Q-value functions to mitigate bias, and it adopts a soft policy evaluation step to update the critic networks, ensuring more stable and efficient learning.

## License

This project is licensed under the CC-BY [License](https://creativecommons.org/licenses/by/4.0/).
