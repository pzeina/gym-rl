# gym-rl

A Gymnasium 2D-environment for agents trained with Reinforcement Learning

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![static analysis workflow](https://github.com/BioDisCo/python-template/actions/workflows/static-analysis.yaml/badge.svg)](https://github.com/BioDisCo/python-template/actions/workflows/static-analysis.yaml/)
[![test workflow](https://github.com/BioDisCo/python-template/actions/workflows/test.yaml/badge.svg)](https://github.com/BioDisCo/python-template/actions/workflows/test.yaml/)

# Usage

```bash
python debug/test_imports.py && echo "✅ Ready to train!"
```

ssh -L 16006:127.0.0.1:6006 olivier@my_server_ip


Prepare the environment:
```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In case you encounter an issue with box2d, like `Failed to build box2d-py`, you can try installing the necessary system dependencies with:

For Ubuntu/Debian:
```shell
sudo apt update
sudo apt install -y python3-dev python3-pip build-essential swig
```

For MacOS:
```shell
brew install swig
```

Train the agent:
```shell
python train.py
```

Or in the background:
```shell
nohup python train.py > model/mylog.txt 2>&1 &
```


# Gymnasium Examples
Some simple examples of Gymnasium environments and wrappers.
For some explanations of these examples, see the [Gymnasium documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).


## Installation

To install your new environment, run the following commands:

```{shell}
cd mili_env
pip install -e .
```

## Training

 A few suggestions:

   1. Increase epsilon, the exploration rate. This will cause the agent to take random actions more often, preventing it from getting stuck. You want epsilon to decay over time as the agent learns, but keep it high enough during training.

   2. Implement epsilon-greedy action selection instead of just epsilon-random. With epsilon-greedy, the agent chooses the action with the highest Q-value (what it thinks is the optimal action) (1-epsilon) % of the time. The remaining epsilon% of the time it chooses a random action. This balances exploitation of current knowledge and exploration of new options.

   3. Use entropy bonuses or weight decay on Q-values. This nudges the agent to consider multiple good options instead of settling on just one suboptimal choice.

   4. Consider using a reward shaping function to incentivize the agent taking actions that lead it to less explored states. This stealthily guides the agent's exploration without directly forcing random actions.

Exploration is key to deep reinforcement learning, so getting this right will help your agent reach its full potential. Good luck! 