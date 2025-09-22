# Unit Tests for Multi-Agent Environment and RLlib Integration

This folder contains comprehensive unit tests for all agent methods and functionality in the multi-agent terrain world environment, plus extensive tests for the RLlib integration.

## Test Coverage

### Original Environment Tests

#### 1. Agent Creation Tests (`test_agent_creation.py`)
- Single agent creation and initialization
- Multi-agent environment setup
- Agent attribute validation
- Grouped vs normal spawning
- Agent positioning and separation
- Target assignment and state consistency

#### 2. Agent Communication Tests (`test_agent_communication.py`)
- Message creation and structure validation
- Message sending and receiving
- Communication range validation
- Agent roles and hierarchy
- Message types and content validation
- Help requests and responses
- Enemy/ally reporting
- Order giving and following
- Communication statistics tracking

#### 3. Agent Movement Tests (`test_agent_movement.py`)
- Action execution (IDLE, FORWARD, BACKWARD, ROTATE_LEFT, ROTATE_RIGHT)
- Movement validation and position updates
- Boundary collision detection
- Energy consumption during movement
- Direction tracking and rotation
- Target direction and distance calculations
- Movement sequence testing
- Action validation and error handling

#### 4. Agent Vision Tests (`test_agent_vision.py`)
- Vision range and communication range validation
- Position and direction methods
- Target-related calculations
- Vision map creation and management
- Vision ray computation
- Game map access and interaction
- Agent state access and consistency
- Known agents tracking system
- Vision attribute existence and initialization

#### 5. Agent Reward Tests (`test_agent_rewards.py`)
- Centralized reward calculation
- Target reaching rewards
- Energy consumption penalties
- Health and survival rewards
- Cooperation reward system
- Individual vs team rewards
- Action-specific reward differences
- Distance-based reward calculations
- Multi-agent reward distribution
- Reward bounds and consistency validation

### RLlib Integration Tests

#### 6. RLlib Wrapper Tests (`test_rllib_wrapper.py`)
- TerrainWorldRLlibWrapper functionality
- Multi-agent environment interface compliance
- Observation and action space validation
- Episode reset and step functionality
- Cooperative reward distribution
- Agent ID management and consistency
- Error handling for invalid inputs
- Memory cleanup and resource management
- Ray integration compatibility

#### 7. RLlib Training Tests (`test_rllib_training.py`)
- Training configuration creation
- Algorithm-specific parameter handling (PPO, DQN, IMPALA, SAC)
- Multi-agent policy setup and sharing
- Callback system functionality
- Device selection and management
- Environment registration with Ray
- Configuration validation and error handling
- Framework and infrastructure setup

#### 8. RLlib Hyperparameter Tuning Tests (`test_rllib_hyperparameters.py`)
- OptimizationConfig creation and validation
- Default hyperparameter retrieval for all algorithms
- Hyperparameter suggestion mechanisms
- Optuna integration and study management
- Objective function structure and execution
- Algorithm-specific parameter ranges
- Optimization direction handling
- Integration testing with Ray Tune

#### 9. RLlib Environment Integration Tests (`test_rllib_environment.py`)
- Environment creation and initialization
- Multi-agent observation and action handling
- Episode lifecycle management
- Reward structure and cooperative behavior
- Termination and truncation conditions
- Different agent count configurations
- Observation and action bounds validation
- State persistence and consistency
- Concurrent environment testing
- Memory management and cleanup

## Running Tests

### Run All Tests
```bash
cd tests
python run_tests.py
```

### Run Only Agent Tests (Original Environment)
```bash
cd tests
python run_tests.py agent
```

### Run Only RLlib Tests
```bash
cd tests
python run_tests.py rllib
```

### Run Individual Test Modules
```bash
cd tests
# Original environment tests
python -m unittest test_agent_creation.py
python -m unittest test_agent_communication.py
python -m unittest test_agent_movement.py
python -m unittest test_agent_vision.py
python -m unittest test_agent_rewards.py

# RLlib integration tests
python -m unittest test_rllib_wrapper.py
python -m unittest test_rllib_training.py
python -m unittest test_rllib_hyperparameters.py
python -m unittest test_rllib_environment.py
```

### Run Specific Test Classes
```bash
python -m unittest test_agent_creation.TestAgentCreation
python -m unittest test_rllib_wrapper.TestRLlibWrapper
python -m unittest test_rllib_training.TestRLlibTraining.test_create_rllib_config_ppo
```

## Test Environment

### Original Environment Tests
The tests use the `TerrainWorldEnv` environment with:
- 2-8 agents depending on test requirements
- No rendering (`render_mode=None`) for faster execution
- Standard environment configuration
- Proper setup and teardown for each test

### RLlib Integration Tests
The RLlib tests use:
- Ray initialized in local mode for testing
- TerrainWorldRLlibWrapper for multi-agent compatibility
- Mocked components for hyperparameter optimization
- Temporary directories for model storage
- Proper Ray shutdown after test completion

## Key Test Features

### Original Environment Tests
1. **Comprehensive Coverage**: Tests cover all major agent functionality including creation, communication, movement, vision, and rewards.
2. **Multi-Agent Focus**: Tests specifically validate multi-agent interactions, cooperation, and communication systems.
3. **Edge Case Testing**: Includes boundary conditions, error handling, and invalid input scenarios.
4. **State Consistency**: Validates that agent states remain consistent across different access methods.

### RLlib Integration Tests
1. **Multi-Agent Compatibility**: Validates that the environment works correctly with RLlib's multi-agent interface.
2. **Algorithm Support**: Tests all supported algorithms (PPO, DQN, IMPALA, SAC) with proper configuration.
3. **Training Infrastructure**: Validates the complete training pipeline including hyperparameter tuning.
4. **Resource Management**: Tests proper initialization and cleanup of Ray and other resources.
5. **Error Handling**: Comprehensive testing of error conditions and invalid configurations.

## Expected Test Results

All tests should pass when the multi-agent environment and RLlib integration are properly configured. The test suite validates:

### Original Environment
- ✅ Agent creation and initialization
- ✅ Communication system functionality  
- ✅ Movement and action execution
- ✅ Vision and detection capabilities
- ✅ Reward calculation and distribution
- ✅ Multi-agent cooperation mechanics
- ✅ Grid-based spawning system

### RLlib Integration
- ✅ Multi-agent wrapper functionality
- ✅ Training configuration and setup
- ✅ Hyperparameter optimization system
- ✅ Environment integration with Ray
- ✅ Cooperative multi-agent training
- ✅ Resource management and cleanup
- ✅ Error handling and validation

## Notes about the environment / RLlib wrapper contract

Recent change: the low-level `TerrainWorldEnv` now follows Gymnasium's single-agent `step()` return convention for compatibility with the passive environment checker. Concretely:

- `env.step(action)` returns `(obs, reward, terminated, truncated, info)` where:
  - `obs` is a dict of per-agent observations (format: `{agent_id: obs}`),
  - `reward` is a scalar `float` representing the centralized team reward,
  - `terminated` and `truncated` are global booleans (True if any agent terminated/truncated),
  - `info` is a dict of per-agent info values.

- The RLlib wrapper `TerrainWorldRLlibWrapper` accepts this format and expands the scalar `reward` and global `terminated`/`truncated` into per-agent dictionaries of the form expected by RLlib (e.g. `{agent_id: reward}` and per-agent dones with the special `"__all__"` key). The wrapper is also robust to older behavior where the env returned per-agent dicts directly.

Why this matters for tests:
- Unit tests that exercise the wrapper and RLlib integration still receive per-agent dictionaries (via the wrapper). Tests that call the low-level env directly should expect the scalar/global returns described above.

Running the focused RLlib wrapper test (example):
```bash
pytest tests/test_rllib_wrapper.py::TestRLlibWrapper::test_cooperative_rewards -q
```

## Troubleshooting

If tests fail, check:

### Common Issues
1. Environment dependencies are installed (ray, optuna, torch)
2. Python path includes the project root
3. All required modules are accessible
4. Environment configuration is correct

### RLlib-Specific Issues
1. Ray is properly installed and can initialize
2. GPU/CUDA availability matches test expectations
3. Temporary directory permissions are correct
4. No port conflicts with Ray services

### Memory Issues
1. Ray workers are properly shut down after tests
2. Environment instances are properly cleaned up
3. Sufficient memory available for multi-agent environments

## Test Statistics

The complete test suite includes approximately 120+ individual test methods covering:

### Original Environment Tests (60+ tests)
- Agent creation and spawning: 8 tests
- Communication system: 12 tests  
- Movement and actions: 13 tests
- Vision and detection: 15 tests
- Reward systems: 12 tests

### RLlib Integration Tests (60+ tests)
- Multi-agent wrapper: 25 tests
- Training configuration: 15 tests
- Hyperparameter tuning: 15 tests
- Environment integration: 20 tests

Total test coverage ensures robust validation of both the original multi-agent environment and the complete RLlib integration for cooperative multi-agent reinforcement learning.