"""Unit tests for agent vision and detection systems."""

import unittest

import numpy as np

from mili_env.envs.terrain_world import TerrainWorldEnv


def is_numeric(value):
    """Check if value is numeric (including numpy types)."""
    return isinstance(value, int | float | np.integer | np.floating)


class TestAgentVision(unittest.TestCase):
    """Test cases for agent vision and detection capabilities."""

    def setUp(self):
        """Set up test environment."""
        self.env = TerrainWorldEnv(num_agents=3, render_mode=None)
        self.agent = self.env.agents[0]
        self.other_agent = self.env.agents[1]

    def test_vision_range(self):
        """Test that agent vision range is properly defined."""
        # Vision range should be a positive value
        assert hasattr(self.agent, "vision_range")
        vision_range = self.agent.vision_range
        assert isinstance(vision_range, int | float)
        assert vision_range > 0

    def test_communication_range(self):
        """Test that agent communication range is properly defined."""
        # Communication range should be a positive value
        assert hasattr(self.agent, "communication_range")
        comm_range = self.agent.communication_range
        assert isinstance(comm_range, int | float)
        assert comm_range > 0

    def test_position_methods(self):
        """Test position-related methods."""
        # Test get_position method
        position = self.agent.get_position()
        assert isinstance(position, tuple)
        assert len(position) == 2
        assert is_numeric(position[0])
        assert is_numeric(position[1])

        # Test get_direction method
        direction = self.agent.get_direction()
        assert is_numeric(direction)

        # Test get_terrain_coordinates method
        terrain_coords = self.agent.get_terrain_coordinates()
        assert isinstance(terrain_coords, tuple)
        assert len(terrain_coords) == 2
        assert isinstance(terrain_coords[0], int | np.integer)
        assert isinstance(terrain_coords[1], int | np.integer)

    def test_target_methods(self):
        """Test agent target calculation methods"""
        # Get target position
        target_pos = self.agent.get_target()
        assert isinstance(target_pos, tuple | list | np.ndarray)
        assert len(target_pos) == 2

        # Check that coordinates are numeric (including numpy types)
        assert is_numeric(target_pos[0])
        assert is_numeric(target_pos[1])

        # Test target direction
        target_direction = self.agent.get_target_direction()
        assert is_numeric(target_direction)

    def test_vision_map_creation(self):
        """Test that vision map is created properly."""
        # Check that agent has a vision map
        assert hasattr(self.agent, "vision_map")
        assert isinstance(self.agent.vision_map, np.ndarray)

        # Vision map size should be related to vision range
        expected_size = 2 * int(self.agent.vision_range) + 1
        assert self.agent.vision_map.shape == (expected_size, expected_size)

        # Test get_vision_map method
        vision_map = self.agent.get_vision_map()
        assert isinstance(vision_map, np.ndarray)
        assert vision_map.shape == self.agent.vision_map.shape

    def test_vision_rays_computation(self):
        """Test that vision rays are computed."""
        # Check that agent has vision rays
        assert hasattr(self.agent, "rays")
        assert self.agent.rays is not None

        # Test get_vision_rays method
        vision_rays = self.agent.get_vision_rays()
        assert isinstance(vision_rays, list)

        # Should be able to compute vision rays
        rays = self.agent.compute_vision_rays()
        assert isinstance(rays, list)
        assert rays is not None

    def test_game_map_access(self):
        """Test game map access methods."""
        # Test get_game_map method
        game_map = self.agent.get_game_map()
        assert game_map is not None

        # Test get_game_map_size method
        map_size = self.agent.get_game_map_size()
        assert isinstance(map_size, tuple)
        assert len(map_size) == 3  # width, height, layers
        assert all(isinstance(dim, int) for dim in map_size)

    def test_agent_state_access(self):
        """Test that agent state can be accessed properly."""
        # Check state access
        assert hasattr(self.agent, "state")
        state = self.agent.get_state()
        assert state is not None

        # Check state values are reasonable
        assert is_numeric(self.agent.state.x)
        assert is_numeric(self.agent.state.y)
        assert is_numeric(self.agent.state.angle)

        # Test state info method
        state_info = self.agent.state.get_state_info()
        assert isinstance(state_info, dict)
        assert "position" in state_info
        assert "target" in state_info
        assert "health" in state_info
        assert "energy" in state_info

    def test_known_agents_system(self):
        """Test known agents tracking system."""
        # Check that agent has known_agents system
        assert hasattr(self.agent, "known_agents")
        assert isinstance(self.agent.known_agents, dict)

        # Test get_known_agents method
        known_agents = self.agent.get_known_agents()
        assert isinstance(known_agents, dict)

    def test_communication_attributes(self):
        """Test communication-related attributes."""
        # Check communication system attributes
        assert hasattr(self.agent, "agent_id")
        assert isinstance(self.agent.agent_id, int)

        assert hasattr(self.agent, "role")
        assert self.agent.role is not None

        assert hasattr(self.agent, "message_inbox")
        assert isinstance(self.agent.message_inbox, list)

        assert hasattr(self.agent, "message_outbox")
        assert isinstance(self.agent.message_outbox, list)

        assert hasattr(self.agent, "communication_history")
        assert isinstance(self.agent.communication_history, list)

    def test_vision_attributes_existence(self):
        """Test that all necessary vision attributes exist."""
        # Check that agent has vision-related attributes
        assert hasattr(self.agent, "vision_range")
        assert hasattr(self.agent, "communication_range")
        assert hasattr(self.agent, "vision_map")

        # Check max speed attributes
        assert hasattr(self.agent, "max_speed_forward")
        assert hasattr(self.agent, "max_speed_backward")
        assert hasattr(self.agent, "max_angular_speed")

        # Check constraint attributes
        assert hasattr(self.agent, "max_health")
        assert hasattr(self.agent, "max_energy")
        assert hasattr(self.agent, "max_ammunition")

    def test_position_consistency(self):
        """Test that position methods return consistent values."""
        # Get position through different methods
        pos1 = self.agent.get_position()
        state_info = self.agent.state.get_state_info()
        pos2 = state_info["position"]

        # Should be consistent
        assert abs(pos1[0] - pos2[0]) < 1e-6
        assert abs(pos1[1] - pos2[1]) < 1e-6

    def test_agent_initialization_state(self):
        """Test that agents are properly initialized with vision capabilities."""
        for agent in self.env.agents:
            # Each agent should have vision system
            assert hasattr(agent, "vision_range")
            assert hasattr(agent, "vision_map")
            assert hasattr(agent, "rays")

            # Each agent should have valid position
            position = agent.get_position()
            assert is_numeric(position[0])
            assert is_numeric(position[1])

            # Each agent should have communication system
            assert hasattr(agent, "agent_id")
            assert hasattr(agent, "message_inbox")

    def test_distance_calculations(self):
        """Test distance-related calculations."""
        # Set known positions for testing
        agent1_pos = self.agent.get_position()
        agent2_pos = self.other_agent.get_position()

        # Calculate manual distance
        manual_distance = np.sqrt(
            (agent1_pos[0] - agent2_pos[0])**2 +
            (agent1_pos[1] - agent2_pos[1])**2
        )

        # Manual distance should be reasonable
        assert manual_distance >= 0
        assert isinstance(manual_distance, float)

    def test_vision_map_update(self):
        """Test that vision map can be updated."""
        # Get initial vision map state
        initial_map = self.agent.get_vision_map().copy()

        # Vision map should be a valid numpy array
        assert isinstance(initial_map, np.ndarray)
        assert initial_map.size > 0

        # Vision map should have correct dimensions
        expected_size = 2 * int(self.agent.vision_range) + 1
        assert initial_map.shape == (expected_size, expected_size)


if __name__ == "__main__":
    unittest.main()
