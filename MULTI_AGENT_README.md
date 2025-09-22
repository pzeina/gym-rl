# Multi-Agent Terrain World Environment

This document describes the multi-agent cooperative version of the TerrainWorldEnv, transformed from a single-agent environment to support decentralized multi-agent reinforcement learning with centralized rewards.

## Overview

The multi-agent TerrainWorldEnv supports:
- **Decentralized Observations**: Each agent observes only its local state
- **Decentralized Training**: Each agent has its own policy
- **Centralized Reward**: Shared reward promotes cooperative behavior
- **Cooperative Goals**: All agents must reach their respective targets efficiently

## Key Features

### 🤖 Multi-Agent Support
- Configurable number of agents (1-8 agents supported)
- Each agent has its own target zone
- Agents are visually distinguished with different colors
- Minimum spawn distance enforced between agents

### 👁️ Decentralized Observations
Each agent receives its own observation dictionary containing:
```python
{
    "position": np.array([x, y]),           # Agent's current position
    "target_position": np.array([tx, ty]),  # Agent's target position
    "distance": np.array([dist]),           # Distance to target
    "direction": np.array([angle]),         # Agent's current orientation
    "target_direction": np.array([t_angle]), # Direction to target
    "energy": np.array([energy]),           # Agent's energy level
    "agent_id": int                         # Agent's unique identifier
}
```

### 🎯 Decentralized Actions
Each agent can take independent actions:
- `IDLE` (0): No movement
- `FORWARD` (1): Move forward
- `BACKWARD` (2): Move backward  
- `ROTATE_LEFT` (3): Turn left
- `ROTATE_RIGHT` (4): Turn right

### 🏆 Centralized Reward System
All agents receive the same cooperative reward based on:

#### Mission Completion (+1000 points)
- Bonus when ALL agents reach their targets
- Additional energy and health preservation bonuses

#### Mission Failure (-500 points)
- Penalty when ANY agent dies or runs out of energy

#### Progress Rewards (continuous)
- **Distance Progress**: +10 points per unit of collective progress toward targets
- **Cooperation Bonus**: +5 points for agent pairs within communication range
- **Energy Efficiency**: +5 points for maintaining high energy levels

#### Cooperation Mechanics
- Agents are rewarded for staying within communication range of each other
- Penalty for agents that are too far apart (> 2x communication range)
- Promotes formation flying and team coordination

## Usage

### Basic Usage

```python
from mili_env.envs.terrain_world import TerrainWorldEnv

# Create environment with 3 agents
env = TerrainWorldEnv(
    num_agents=3,
    target_zone_size=15,
    terrain_filename="plain_terrain.csv"
)

# Reset environment
observations, info = env.reset(seed=42)

# Create actions for all agents
actions = {
    "agent_0": 1,  # FORWARD
    "agent_1": 3,  # ROTATE_LEFT
    "agent_2": 1,  # FORWARD
}

# Step environment
observations, rewards, terminated, truncated, info = env.step(actions)
```

### Training with Stable-Baselines3

The environment is compatible with multi-agent training libraries. Here's an example structure:

```python
# Pseudo-code for multi-agent training
for agent_id in range(num_agents):
    # Each agent has its own policy
    policies[f"agent_{agent_id}"] = PPO(
        policy="MultiInputPolicy",
        env=env,
        # ... other parameters
    )

# Training loop
for episode in range(num_episodes):
    observations = env.reset()
    
    while not done:
        actions = {}
        for agent_id in range(num_agents):
            # Each agent selects action based on its local observation
            agent_key = f"agent_{agent_id}"
            action = policies[agent_key].predict(observations[agent_key])
            actions[agent_key] = action
        
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        # All agents receive the same centralized reward
        for agent_id in range(num_agents):
            agent_key = f"agent_{agent_id}"
            policies[agent_key].learn(
                obs=observations[agent_key],
                reward=rewards[agent_key],  # Same for all agents
                # ... other parameters
            )
```

## Environment Spaces

### Observation Space
```python
gym.spaces.Dict({
    "agent_0": gym.spaces.Dict({
        "position": gym.spaces.Box(0, max_dim-1, shape=(2,), dtype=np.float64),
        "target_position": gym.spaces.Box(0, max_dim-1, shape=(2,), dtype=np.int64),
        "distance": gym.spaces.Box(0.0, max_distance, shape=(1,), dtype=np.float64),
        "direction": gym.spaces.Box(0.0, 2*π, shape=(1,), dtype=np.float64),
        "target_direction": gym.spaces.Box(0.0, 2*π, shape=(1,), dtype=np.float64),
        "energy": gym.spaces.Box(0.0, 100.0, shape=(1,), dtype=np.float64),
        "agent_id": gym.spaces.Discrete(num_agents),
    }),
    # ... repeat for each agent
})
```

### Action Space
```python
gym.spaces.Dict({
    "agent_0": gym.spaces.Discrete(5),  # 5 possible actions
    "agent_1": gym.spaces.Discrete(5),
    # ... repeat for each agent
})
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_agents` | int | 1 | Number of agents (1-8) |
| `target_zone_size` | int | 20 | Size of target zones |
| `terrain_filename` | str | "plain_terrain.csv" | Terrain map file |
| `render_mode` | str | None | "human" or "rgb_array" |

## Key Implementation Details

### Agent Initialization
- Agents spawn at random positions outside all target zones
- Minimum distance enforced between agents at spawn
- Each agent gets a unique target zone
- Different colors assigned for visual identification

### Energy Management
- Energy consumption varies by action and terrain
- Movement actions consume more energy on difficult terrain
- Rotation and idle actions have fixed energy costs
- Energy depletion leads to mission failure

### Termination Conditions
- **Success**: All agents reach their targets
- **Failure**: Any agent dies (health ≤ 0) or runs out of energy
- **Global termination**: Episode ends when any termination condition is met

### Reward Design Philosophy
The centralized reward system encourages:
1. **Individual Progress**: Each agent moving toward its target
2. **Team Coordination**: Agents staying close for communication
3. **Resource Efficiency**: Preserving energy and health
4. **Mission Success**: Completing objectives as a team

## Example Output

When running the multi-agent environment:

```
Multi-Agent Terrain World Environment
=====================================
Number of agents: 3
Map size: 50 x 50

Initial State:
==============

Agent 0:
  Position: [12.0, 34.0]
  Target: [45, 15]
  Distance to target: 38.00
  Energy: 100.00

Agent 1:
  Position: [8.0, 12.0]
  Target: [25, 40]
  Distance to target: 45.00
  Energy: 100.00

Agent 2:
  Position: [35.0, 8.0]
  Target: [10, 35]
  Distance to target: 52.00
  Energy: 100.00

Running simulation...
=====================

Step 1:
Centralized reward: 2.15
  Agent 0 - Distance to target: 37.20, Energy: 99.85
  Agent 1 - Distance to target: 44.15, Energy: 99.85
  Agent 2 - Distance to target: 51.30, Energy: 99.85
```

## Research Applications

This environment is suitable for research in:
- **Multi-Agent Reinforcement Learning (MARL)**
- **Decentralized Cooperative Learning**
- **Communication and Coordination**
- **Formation Control**
- **Multi-Robot Navigation**
- **Swarm Intelligence**

## Comparison: Single-Agent vs Multi-Agent

| Aspect | Single-Agent | Multi-Agent |
|--------|--------------|-------------|
| Observations | Global state | Local state per agent |
| Actions | Single action | Dict of actions |
| Rewards | Individual reward | Centralized team reward |
| Termination | Individual success/failure | Team success/failure |
| Complexity | Linear scaling | Exponential scaling |
| Cooperation | N/A | Explicitly rewarded |

The multi-agent version maintains all the complexity and richness of the original environment while adding the challenges and opportunities of multi-agent coordination.