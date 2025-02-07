import signal
import sys
import gymnasium as gym
import numpy as np
from mili_env.envs.classes.qlearning_agent import QLearningAgent, AgentConfig
from mili_env.envs.visualization import GradientLossVisualization
from mili_env.envs.classes.robot_base import Actions
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm
from pathlib import Path
import torch

def save_periodic_model(agent, episode, base_model_path) -> None:
    """Save the model to a temporary file and replace the main model file."""
    temp_model_path = base_model_path.with_name(f"{base_model_path.stem}_temp{base_model_path.suffix}")
    
    # Ensure directory exists
    temp_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to temporary file
    agent.save_model(temp_model_path)
    
    # Replace the main model file
    temp_model_path.replace(base_model_path)
    
    print(f"Model saved at episode {episode}")

# Hyperparameters
n_episodes = 10_000
start_epsilon = 1.0
config = AgentConfig(
    learning_rate=0.01,
    initial_epsilon=1.0,
    epsilon_decay=start_epsilon / (n_episodes / 2),  # reduce the exploration over time
    final_epsilon=0.1,
    discount_factor=0.95,
    memory_size=100_000,
    batch_size=64,
    hidden_size=256
)

model_path = Path("model/qlearning_model.pth")
env_terrain = gym.make("mili_env/TerrainWorld-v0", render_mode="rgb_array")
env = RecordEpisodeStatistics(env_terrain, buffer_length=n_episodes)

visualization = GradientLossVisualization(128, 128, 256)
agent = QLearningAgent(env, config, visualization)

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent.model.to(device)

# Load the model if it exists and is valid
if model_path.exists():
    try:
        agent.load_model(model_path)
        agent.model.to(device)  # Ensure the model is moved to the correct device after loading
    except Exception as e:
        print(f"Failed to load model: {e}")

# Handle interruptions gracefully
def save_model_and_exit(signum, frame):
    save_periodic_model(agent, -1, model_path)
    sys.exit(0)

signal.signal(signal.SIGINT, save_model_and_exit)
signal.signal(signal.SIGTERM, save_model_and_exit)

# Training loop
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    terminated = False
    
    # Play one episode
    while not terminated:
        state = np.array(list(obs["position"].flatten()) + list(obs["direction"].flatten()) + list(obs["target"].flatten()))
        action: Actions = Actions(agent.get_action(state))
        next_obs, reward, terminated, win, info = env.step(action)
        next_state = np.array(list(next_obs["position"].flatten()) + list(next_obs["direction"].flatten()) + list(next_obs["target"].flatten()))

        # Remember the experience
        agent.remember(state, action.value, reward, next_state, terminated)
        
        # Train short memory
        agent.train_short_memory(state, action.value, reward, next_state, terminated)
        
        # Update observations
        obs = next_obs
    
    # Train long memory
    agent.train_long_memory()
    
    # Decay epsilon
    agent.decay_epsilon()
    
    # Save periodically
    if (episode + 1) % 100 == 0:
        save_periodic_model(agent, episode, model_path)

# Save the final model
save_periodic_model(agent, n_episodes, model_path)