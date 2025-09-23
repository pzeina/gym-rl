# gym-rl

A Gymnasium 2D-environment for agents trained with Reinforcement Learning

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![static analysis workflow](https://github.com/BioDisCo/python-template/actions/workflows/static-analysis.yaml/badge.svg)](https://github.com/BioDisCo/python-template/actions/workflows/static-analysis.yaml/)
[![test workflow](https://github.com/BioDisCo/python-template/actions/workflows/test.yaml/badge.svg)](https://github.com/BioDisCo/python-template/actions/workflows/test.yaml/)


# Usage

## Preliminary Check

To verify that your environment is correctly set up, run the following command from the root of the repository:
```bash
for f in debug/*.py; do  python $f || exit 1; done && python debug/test_imports.py && echo "✅ Ready to train!"
```

## Via SSH Tunneling
If you are running the training on a remote server and want to visualize the training process using TensorBoard, you can set up SSH port forwarding upon connecting to the server:
```bash
ssh -L 16006:127.0.0.1:6006 ai
```

Then, start TensorBoard on the remote server:
```bash
tensorboard --logdir models/tb --bind_all
```

## Running the Training Script

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


## Environment / RLlib wrapper contract

Note: To maintain compatibility with Gymnasium's passive environment checker the low-level `TerrainWorldEnv` now returns a Gymnasium-style `step()` 5-tuple:

- `(obs, reward, terminated, truncated, info)` where `reward` is a scalar `float` and `terminated`/`truncated` are global booleans. Observations and `info` remain per-agent dictionaries keyed by agent id.

The `TerrainWorldRLlibWrapper` converts that scalar/global return at the environment boundary into the per-agent dictionaries expected by RLlib (e.g. `{agent_id: reward}` and per-agent dones with the special `"__all__"` key). The wrapper is tolerant of older behavior where the low-level env returned per-agent dicts directly.

Running the focused RLlib wrapper test (example):
```bash
pytest tests/test_rllib_wrapper.py::TestRLlibWrapper::test_cooperative_rewards -q
```

## Reward functions and experiments

This project supports modular reward functions for the multi-agent `TerrainWorldEnv`.

Register reward functions in Python and select them at runtime:

```python
from mili_env.envs.terrain_world import TerrainWorldEnv

env = TerrainWorldEnv(num_agents=3)

def per_agent(prev_info):
  # return dict like {"agent_0": 1.0, ...}
  ...

def team(prev_info, per_agent_rewards):
  # return a scalar team reward
  ...

env.register_reward_function("my_reward", per_agent_fn=per_agent, team_fn=team)
env.set_reward_function("my_reward")
env.set_reward_mode("team")  # or "per_agent" or "both"
```

Archival and provenance

 - Each `reset()` will append a compact JSON line to `debug/reward_runs.jsonl` with the selected reward name, mode, number of agents and seed (if provided).
 - Use `env.archive_reward_function_details(name)` to append the function's docstring and best-effort source to `debug/reward_functions.jsonl` for reproducibility.

Experiment runner

There is a small helper script at `scripts/run_reward_experiments.py` that iterates over registered reward functions, runs a short number of episodes for each, and writes `debug/reward_experiments.jsonl` with the average reward and metadata. Run it from the repository root:

```bash
python scripts/run_reward_experiments.py
```

The experiment runner will also archive reward function docstrings/source for provenance.