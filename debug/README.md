# Debug Scripts

This directory contains test and debug scripts used during the SB3 RL Tips implementation. You can run these scripts to validate different components of the training system.

## Scripts Overview

### 🔧 `test_environment.py`
Tests the environment creation and basic functionality.

**What it tests:**
- `make_env()` function from `train.py`
- Environment reset and step operations
- Monitor wrapper functionality
- TerrainWorld environment properties

**Usage:**
```bash
python debug/test_environment.py
```

### 🧪 `test_sb3_components.py`
Tests the complete SB3 implementation including all major components.

**What it tests:**
- VecNormalize wrapper functionality
- Parallel environments (DummyVecEnv vs SubprocVecEnv)
- Custom network architecture with policy_kwargs
- Complete training pipeline
- TerrainWorld environment integration

**Usage:**
```bash
python debug/test_sb3_components.py
```

### 🏗️ `test_network_architecture.py`
Specifically tests network architecture configurations and validates the modern format.

**What it tests:**
- Old vs new `net_arch` format (shows deprecation fix)
- Different activation functions (Tanh, ReLU, LeakyReLU)
- Various network sizes and configurations
- Shared vs separate policy/value networks
- Exact configuration from `train.py`

**Usage:**
```bash
python debug/test_network_architecture.py
```

### 📦 `test_imports.py`
Validates all imports and dependencies are correctly installed.

**What it tests:**
- Basic Python library imports
- Stable-Baselines3 imports
- PyTorch imports and versions
- Gymnasium imports
- Custom environment registration
- `train.py` and `utils.py` imports
- GPU availability (CUDA/MPS)

**Usage:**
```bash
python debug/test_imports.py
```

## Quick Test All

To run all tests quickly:

```bash
cd /Users/pzeinaty/Documents/new-gym
python debug/test_imports.py
python debug/test_environment.py  
python debug/test_network_architecture.py
python debug/test_sb3_components.py
```

## What These Scripts Validate

These debug scripts ensure that all the **SB3 RL Tips recommendations** have been properly implemented:

✅ **Observation Normalization** - VecNormalize with `norm_obs=True`  
✅ **Reward Normalization** - VecNormalize with `norm_reward=True`  
✅ **Parallel Environments** - SubprocVecEnv for better sample efficiency  
✅ **Tuned Hyperparameters** - Learning rate, batch size, clip range, etc.  
✅ **Custom Network Architecture** - Separate policy/value networks with Tanh activation  
✅ **Proper Evaluation** - Deterministic evaluation environment  
✅ **Environment Monitoring** - Monitor wrapper for episode statistics  
✅ **Modern API Usage** - Fixed deprecation warnings in net_arch format  

## Troubleshooting

If any script fails:

1. **Import errors**: Run `pip install -r requirements.txt`
2. **Environment not found**: The TerrainWorld environment may not be registered
3. **GPU warnings**: Normal if CUDA/MPS not available, training will use CPU
4. **Multiprocessing issues**: SubprocVecEnv may fall back to DummyVecEnv on some systems

## Implementation Context

These scripts were created during the implementation of recommendations from:
https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

The goal was to transform a basic PPO training script into a production-ready implementation following research-backed best practices from the SB3 team.