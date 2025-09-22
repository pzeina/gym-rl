from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import ray
import torch
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.impala import IMPALAConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

import mili_env  # ensure the package is imported so Gymnasium sees registered envs  # noqa: F401
import utils
from hyperparameter_tuning_rllib import OptimizationConfig, get_default_hyperparams, tune_hyperparameters_rllib

logger = logging.getLogger(__name__)


class ArgsProtocol(Protocol):
    """Protocol for command line arguments."""
    algorithm: str
    parallel: int | None
    device: str
    tune_hyperparams: bool | None = None
    study_name: str | None = None
    n_trials: int | None = None


class TerrainWorldRLlibWrapper(MultiAgentEnv):
    """RLlib wrapper for TerrainWorldEnv to handle multi-agent interface."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the wrapper with the underlying environment."""
        import gymnasium as gym
        import numpy as np

        # IMPORTANT: Ensure Gymnasium sees custom env registrations inside each Ray worker.
        # In remote actors, top-level imports of this module may not execute before class
        # construction (cloudpickle can deserialize classes without re-importing the module).
        # Explicitly import the package here so its gym.register() calls run in this process.
        try:
            import mili_env as _  # noqa: F401
        except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover - best-effort safety
            # Proceed; gym.make below will raise if registration is still missing.
            logger.warning("Failed to import mili_env in worker: %s", e)

        # Extract environment configuration
        env_config = config.get("env_config", {}) if config else {}
        num_agents = env_config.get("num_agents", 4)
        render_mode = env_config.get("render_mode", None)

        # Create the underlying environment
        self.env = gym.make(
            "mili_env/TerrainWorld-v0",
            num_agents=num_agents,
            render_mode=render_mode
        )

        # Extract agent IDs and validate spaces
        act_space = self.env.action_space
        obs_space = self.env.observation_space
        if not (isinstance(act_space, gym.spaces.Dict) and isinstance(obs_space, gym.spaces.Dict)):
            msg = "Underlying env must provide Dict action and observation spaces"
            raise TypeError(msg)

        self.agent_ids = list(act_space.spaces.keys())
        self._num_agents = len(self.agent_ids)
        if self._num_agents == 0:
            msg = "No agents found in environment"
            raise ValueError(msg)

        # For DQN compatibility: flatten Dict observation space to 1D Box
        # Each agent observes: position(2) + target_position(2) + distance(1) +
        # direction(1) + target_direction(1) + energy(1) = 8 features
        self._obs_dim = 8
        per_agent_box = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float64
        )

        # Expose mapping-style spaces expected by RLlib's vector multi-agent wrappers
        self.observation_space = {aid: per_agent_box for aid in self.agent_ids}  # type: ignore[assignment]
        self.action_space = {aid: act_space.spaces[aid] for aid in self.agent_ids}  # type: ignore[assignment]

        # Cache dimensions for normalization
        self._width = getattr(self.env, "width", None)
        self._height = getattr(self.env, "height", None)
        self._max_dist = (
            float(np.sqrt(float(self._width) ** 2 + float(self._height) ** 2))
            if self._width and self._height
            else 1.0
        )

    def _process_obs(self, obs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Flatten per-agent Dict observations to 1D float32 arrays for DQN compatibility."""
        import numpy as np

        flattened: dict[str, Any] = {}
        for aid, aobs in obs.items():
            # Extract observation components (skip agent_id for DQN compatibility)
            pos = np.asarray(aobs.get("position", [0, 0]), dtype=np.float32)
            target_pos = np.asarray(aobs.get("target_position", [0, 0]), dtype=np.float32)
            distance = np.asarray(aobs.get("distance", [0]), dtype=np.float32).reshape(-1)
            direction = np.asarray(aobs.get("direction", [0]), dtype=np.float32).reshape(-1)
            target_direction = np.asarray(aobs.get("target_direction", [0]), dtype=np.float32).reshape(-1)
            energy = np.asarray(aobs.get("energy", [0]), dtype=np.float32).reshape(-1)

            # Normalize values to [0, 1] range for better training stability
            if self._width and self._height:
                pos = np.clip(pos / np.array([self._width, self._height], dtype=np.float32), 0, 1)
                target_pos = np.clip(target_pos / np.array([self._width, self._height], dtype=np.float32), 0, 1)
                distance = np.clip(distance / self._max_dist, 0, 1)
            else:
                # Fallback normalization
                pos = np.clip(pos / 100.0, 0, 1)
                target_pos = np.clip(target_pos / 100.0, 0, 1)
                distance = np.clip(distance / 100.0, 0, 1)

            # Normalize angles to [0, 1] (angles are typically in [0, 2π])
            direction = np.clip(direction / (2 * np.pi), 0, 1)
            target_direction = np.clip(target_direction / (2 * np.pi), 0, 1)

            # Normalize energy (assuming max energy is 100)
            energy = np.clip(energy / 100.0, 0, 1)

            # Concatenate all features into a single 1D array
            flattened_obs = np.concatenate([
                pos, target_pos, distance, direction, target_direction, energy
            ], axis=0).astype(np.float32)

            # Ensure we have exactly the expected number of features
            if len(flattened_obs) != self._obs_dim:
                # Pad or truncate to match expected dimension
                if len(flattened_obs) < self._obs_dim:
                    padding = np.zeros(self._obs_dim - len(flattened_obs), dtype=np.float32)
                    flattened_obs = np.concatenate([flattened_obs, padding])
                else:
                    flattened_obs = flattened_obs[:self._obs_dim]

            # Final clipping to ensure values are strictly within [0, 1]
            flattened_obs = np.clip(flattened_obs, 0.0, 1.0)

            flattened[aid] = flattened_obs

        return flattened

    @property
    def num_agents(self) -> int:
        """Return the number of agents."""
        return self._num_agents

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Reset the environment and return initial observations."""
        obs, info = self.env.reset(seed=seed, options=options)
        # Cast per-agent observations
        proc_obs = self._process_obs(obs)
        return proc_obs, info

    def step(
        self, action_dict: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Step the environment with actions from all agents."""
        # Validate that all agents have actions
        missing_agents = set(self.agent_ids) - set(action_dict.keys())
        if missing_agents:
            msg = f"Missing actions for agents: {missing_agents}"
            raise KeyError(msg)

        # action_dict is already in {agent_id: action} format
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)

        # Normalize rewards: env may return per-agent dict or scalar reward
        if isinstance(rewards, dict):
            per_agent_rewards = rewards
        else:
            # Scalar reward: replicate for all agents
            per_agent_rewards = {agent_id: float(rewards) for agent_id in self.agent_ids}

        # Normalize terminated/truncated: env may return per-agent dicts or booleans
        if isinstance(terminated, dict):
            per_agent_terminated = terminated
        else:
            per_agent_terminated = {agent_id: bool(terminated) for agent_id in self.agent_ids}

        if isinstance(truncated, dict):
            per_agent_truncated = truncated
        else:
            per_agent_truncated = {agent_id: bool(truncated) for agent_id in self.agent_ids}

        # RLlib requires "__all__" key to indicate if all agents are done
        per_agent_terminated["__all__"] = all(per_agent_terminated.values())
        per_agent_truncated["__all__"] = all(per_agent_truncated.values())

        # Cast/clean observations
        proc_obs = self._process_obs(obs)
        return proc_obs, per_agent_rewards, per_agent_terminated, per_agent_truncated, info

    def get_agent_ids(self) -> set[str]:
        """Return the list of agent IDs."""
        return set(self.agent_ids)


# Register the environment at module level
register_env("terrain_world_rllib", TerrainWorldRLlibWrapper)


class TrainingCallbacks(DefaultCallbacks):
    """Custom callbacks for training monitoring and evaluation."""

    def on_episode_step(
        self,
        *,
        worker: object = None,
        base_env: object = None,
        policies: object = None,
        episode: object,
        **kwargs: object
    ) -> None:
        """Called on each episode step."""
        # Log multi-agent metrics
        _ = worker, base_env, policies, kwargs  # Unused but required by interface
        if episode and hasattr(episode, "length") and getattr(episode, "length", 0) == 1:
            cm = getattr(episode, "custom_metrics", None)
            agent_rewards = getattr(episode, "agent_rewards", {})
            if isinstance(cm, dict) and isinstance(agent_rewards, dict):
                cm["agents_alive"] = len(agent_rewards)

    def on_episode_end(
        self,
        *,
        worker: object = None,
        base_env: object = None,
        policies: object = None,
        episode: object,
        **kwargs: object
    ) -> None:
        """Called when an episode ends."""
        # Calculate cooperative metrics
        _ = worker, base_env, policies, kwargs  # Unused but required by interface
        if not hasattr(episode, "agent_rewards"):
            return

        agent_rewards = getattr(episode, "agent_rewards", {})
        total_reward = sum(agent_rewards.values()) if isinstance(agent_rewards, dict) else 0.0
        avg_reward = total_reward / len(agent_rewards) if agent_rewards else 0.0

        cm = getattr(episode, "custom_metrics", None)
        if isinstance(cm, dict):
            cm["total_reward"] = float(total_reward)
            cm["avg_agent_reward"] = float(avg_reward)
            cm["episode_length"] = getattr(episode, "length", 0)

        # Log per-agent rewards
        for agent_id, reward in (agent_rewards.items() if isinstance(agent_rewards, dict) else []):
            try:
                if isinstance(cm, dict):
                    cm[f"reward_{agent_id}"] = float(reward)
            except (TypeError, ValueError) as exc:  # pragma: no cover - best-effort metrics
                logger.debug("Skipping metric for %s due to: %s", agent_id, exc)


def create_multiagent_config(num_agents: int) -> dict[str, Any]:  # noqa: ARG001
    """Create multi-agent configuration for RLlib."""

    # Create policy mapping - all agents share the same policy for cooperative learning
    def policy_mapping_fn(agent_id: str, episode: object = None, worker: object = None, **kwargs: object) -> str:
        """Map all agents to shared policy."""
        _ = agent_id, episode, worker, kwargs  # Unused but required by interface
        return "shared_policy"

    # Define the shared policy for all agents
    policies = {
        "shared_policy": PolicySpec(
            policy_class=None,  # Will be inferred from algorithm
            observation_space=None,  # Will be set from environment
            action_space=None,  # Will be set from environment
            config={}
        )
    }

    return {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["shared_policy"],  # Train only the shared policy
    }



if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


def _create_ppo_config(
    env_config: dict[str, Any],
    multiagent_config: dict[str, Any],
    hyperparams: dict[str, Any],
    device: str,
) -> AlgorithmConfig:
    """Create PPO algorithm configuration."""
    # Extract model config for new API stack if present
    model_config = hyperparams.pop("model", {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
    })

    config = PPOConfig()
    config = config.environment(
        env="terrain_world_rllib",
        env_config=env_config,
        disable_env_checking=True,
    )
    config = config.multi_agent(**multiagent_config)
    config = config.framework("torch")
    # Use new API stack with proper rl_module configuration
    config = config.rl_module(model_config=model_config)
    config = config.training(**hyperparams)
    config = config.env_runners(
        num_env_runners=4,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=1,
    )
    config = config.resources(
        num_gpus=1 if torch.cuda.is_available() and device != "cpu" else 0,
    )
    config = config.callbacks(TrainingCallbacks)
    return config.debugging(log_level="INFO")


def _create_dqn_config(
    env_config: dict[str, Any],
    multiagent_config: dict[str, Any],
    hyperparams: dict[str, Any],
    device: str,
) -> AlgorithmConfig:
    """Create DQN algorithm configuration."""
    # Separate model config for new API stack
    model_config = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    }

    config = DQNConfig()
    config = config.environment(
        env="terrain_world_rllib",
        env_config=env_config,
        disable_env_checking=True,
    )
    config = config.multi_agent(**multiagent_config)
    config = config.framework("torch")
    # Use new API stack with proper rl_module configuration
    config = config.rl_module(model_config=model_config)
    config = config.training(**hyperparams)
    config = config.env_runners(
        num_env_runners=2,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=1,
        # Ensure local worker has an environment to infer spaces
        create_env_on_local_worker=True,
    )
    config = config.resources(
        num_gpus=1 if torch.cuda.is_available() and device != "cpu" else 0,
    )
    config = config.callbacks(TrainingCallbacks)
    return config.debugging(log_level="INFO")


def _create_impala_config(
    env_config: dict[str, Any],
    multiagent_config: dict[str, Any],
    hyperparams: dict[str, Any],
    device: str,
) -> AlgorithmConfig:
    """Create IMPALA algorithm configuration."""
    config = IMPALAConfig()
    config = config.environment(
        env="terrain_world_rllib",
        env_config=env_config,
        disable_env_checking=True,
    )
    config = config.multi_agent(**multiagent_config)
    config = config.framework("torch")
    config = config.training(**hyperparams)
    config = config.env_runners(
        num_env_runners=4,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=1,
    )
    config = config.resources(
        num_gpus=1 if torch.cuda.is_available() and device != "cpu" else 0,
    )
    config = config.callbacks(TrainingCallbacks)
    return config.debugging(log_level="INFO")


def _create_sac_config(
    env_config: dict[str, Any],
    multiagent_config: dict[str, Any],
    hyperparams: dict[str, Any],
    device: str,
) -> AlgorithmConfig:
    """Create SAC algorithm configuration."""
    config = SACConfig()
    config = config.environment(
        env="terrain_world_rllib",
        env_config=env_config,
        disable_env_checking=True,
    )
    config = config.multi_agent(**multiagent_config)
    config = config.framework("torch")
    config = config.training(**hyperparams)
    config = config.env_runners(
        num_env_runners=2,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=1,
    )
    config = config.resources(
        num_gpus=1 if torch.cuda.is_available() and device != "cpu" else 0,
    )
    config = config.callbacks(TrainingCallbacks)
    return config.debugging(log_level="INFO")


def create_algorithm_config(
    algorithm: str,
    hyperparams: dict[str, Any] | None = None,
    env_config: dict[str, Any] | None = None,
    device: str = "auto",
) -> AlgorithmConfig:
    """Create RLlib algorithm configuration."""
    if hyperparams is None:
        hyperparams = get_default_hyperparams(algorithm)
    if env_config is None:
        env_config = {}

    # Base configuration
    num_agents = env_config.get("num_agents", 4)

    # Ensure num_agents is not in hyperparams
    if "num_agents" in hyperparams:
        hyperparams = hyperparams.copy()
        del hyperparams["num_agents"]

    # Multi-agent configuration
    multiagent_config = create_multiagent_config(num_agents)

    # Algorithm-specific configurations
    algorithm_lower = algorithm.lower()
    if algorithm_lower == "ppo":
        return _create_ppo_config(env_config, multiagent_config, hyperparams, device)
    if algorithm_lower == "dqn":
        return _create_dqn_config(env_config, multiagent_config, hyperparams, device)
    if algorithm_lower == "impala":
        return _create_impala_config(env_config, multiagent_config, hyperparams, device)
    if algorithm_lower == "sac":
        return _create_sac_config(env_config, multiagent_config, hyperparams, device)

    msg = f"Unsupported algorithm: {algorithm}"
    raise ValueError(msg)


def setup_ray_and_directories(args: object) -> tuple[int, Path]:
    """Initialize Ray and create model directories."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=getattr(args, "parallel", None) or 8,
            num_gpus=1 if torch.cuda.is_available() and getattr(args, "device", "cpu") != "cpu" else 0,
            ignore_reinit_error=True,
        )

    timestamp = int(time.time())
    algorithm = getattr(args, "algorithm", "unknown")
    model_dir = Path(__file__).resolve().parent / "models" / f"rllib_{algorithm}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training results will be saved to: %s", model_dir)
    logger.info("Run 'tensorboard --logdir %s' to monitor training progress", model_dir)

    return timestamp, model_dir


def setup_device(args: object) -> str:
    """Setup and validate the training device."""
    # Device selection
    device_arg = getattr(args, "device", "auto")
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Auto-selected device: %s", device)
    else:
        device = device_arg
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

    logger.info("Using device: %s", device)
    if device.startswith("cuda"):
        logger.info("CUDA device name: %s", torch.cuda.get_device_name())
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("CUDA memory available: %.2f GB", memory_gb)

    return device


def setup_hyperparameters(args: object, env_config: dict[str, Any], device: str) -> dict[str, Any]:
    """Setup hyperparameters either through optimization or defaults."""
    algorithm = getattr(args, "algorithm", "PPO")
    if getattr(args, "tune_hyperparams", False):
        logger.info("Starting hyperparameter optimization for %s", algorithm)

        study_name = getattr(args, "study_name", None) or f"{algorithm}_{int(time.time())}"

        # Create optimization configuration
        config = OptimizationConfig(
            algorithm=algorithm,
            env_config=env_config,
            device=device,
        )

        best_hyperparams = tune_hyperparameters_rllib(
            config=config,
            n_trials=getattr(args, "n_trials", 10),
            study_name=study_name,
        )

        logger.info("Using optimized hyperparameters for final training")
        return best_hyperparams

    logger.info("Using default hyperparameters for %s", algorithm)
    return get_default_hyperparams(algorithm)


def run_training(args: object, config: AlgorithmConfig, model_dir: Path) -> None:
    """Execute the main training loop."""
    algorithm = getattr(args, "algorithm", "PPO")

    # Training configuration - use new API stack metric names
    stop_criteria = {
        "env_runners/num_env_steps_sampled_lifetime": getattr(args, "timesteps", 1_000_000),
        "env_runners/num_episodes_lifetime": getattr(args, "episodes", 5_000),
    }

    # Checkpoint configuration - enable checkpoints for shorter runs
    timesteps = getattr(args, "timesteps", 1000)
    checkpoint_config = tune.CheckpointConfig(
        checkpoint_frequency=max(1, timesteps // 500),  # Checkpoint every 500 timesteps or at least once
        checkpoint_at_end=True,
    )

    logger.info("Starting training with stop criteria: %s", stop_criteria)

    # Run training with Ray Tune
    results = tune.run(
        algorithm.upper(),
        name=f"{algorithm}_multiagent_terrain",
        config=config.to_dict(),
        stop=stop_criteria,
        checkpoint_config=checkpoint_config,
        storage_path=str(model_dir.parent),
        verbose=1,
        progress_reporter=tune.CLIReporter(
            metric_columns=["env_runners/episode_return_mean", "env_runners/num_env_steps_sampled_lifetime"],
            sort_by_metric=True,
        ),
        keep_checkpoints_num=5,  # Keep last 5 checkpoints
        checkpoint_score_attr="env_runners/episode_return_mean",
    )

    # Get the best trial
    best_trial = results.get_best_trial("env_runners/episode_return_mean", "max")
    if best_trial is None:
        logger.error("No trials completed successfully")
        return

    logger.info("Best trial config: %s", best_trial.config)
    logger.info("Best trial result: %s", best_trial.last_result)

    # Save best checkpoint path
    if best_trial.checkpoint and hasattr(best_trial.checkpoint, "path"):
        best_checkpoint = best_trial.checkpoint.path
        logger.info("Best checkpoint saved at: %s", best_checkpoint)
    else:
        best_checkpoint = "No checkpoint available"
        logger.warning("No checkpoint found for best trial")

    # Save training results summary
    results_summary = {
        "algorithm": algorithm,
        "best_reward": (
            best_trial.last_result.get("env_runners/episode_return_mean", 0)
            if best_trial.last_result else 0
        ),
        "timesteps": (
            best_trial.last_result.get("env_runners/num_env_steps_sampled_lifetime", 0)
            if best_trial.last_result else 0
        ),
        "episodes": best_trial.last_result.get("env_runners/num_episodes_lifetime", 0) if best_trial.last_result else 0,
        "best_checkpoint": str(best_checkpoint),
        "config": best_trial.config,
    }

    summary_file = model_dir / "training_summary.json"
    with summary_file.open("w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    best_reward = best_trial.last_result.get("env_runners/episode_return_mean", 0) if best_trial.last_result else 0
    logger.info("Training complete! Best reward: %.2f", best_reward)
    logger.info("Training summary saved to: %s", summary_file)


def main() -> None:
    """Train multi-agent RL using RLlib with cooperative behavior."""
    args = utils.parse_args()

    try:
        # Setup Ray and directories
        timestamp, model_dir = setup_ray_and_directories(args)

        # Register the environment
        register_env("terrain_world_rllib", TerrainWorldRLlibWrapper)

        # Environment configuration
        env_config = {
            "num_agents": 4,  # Default number of agents
            "render_mode": None,
        }

        # Setup device
        device = setup_device(args)

        # Setup hyperparameters
        hyperparams = setup_hyperparameters(args, env_config, device)

        # Create algorithm configuration
        config = create_algorithm_config(
            algorithm=args.algorithm,
            env_config=env_config,
            hyperparams=hyperparams,
            device=device,
        )

        algorithm_name = args.algorithm.upper()
        logger.info("Created %s configuration for multi-agent training", algorithm_name)
        logger.info("Hyperparameters: %s", hyperparams)

        # Run training
        run_training(args, config, model_dir)

    except Exception:
        logger.exception("Training failed")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
