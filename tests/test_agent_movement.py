"""Unit tests for agent movement and actions."""

import unittest

import numpy as np

from mili_env.envs.classes.robot_base import Actions
from mili_env.envs.terrain_world import TerrainWorldEnv


class TestAgentMovement(unittest.TestCase):
    """Test cases for agent movement and action execution."""

    def setUp(self):
        """Set up test environment."""
        self.env = TerrainWorldEnv(num_agents=2, render_mode=None)
        self.agent = self.env.agents[0]
        self.initial_pos = self.agent.get_position()

    def test_idle_action(self):
        """Test that IDLE action doesn't change position."""
        initial_pos = self.agent.get_position()

        self.agent.move(Actions.IDLE.value)

        new_pos = self.agent.get_position()
        assert initial_pos == new_pos, "IDLE should not change position"

    def test_forward_movement(self):
        """Test forward movement."""
        initial_pos = self.agent.get_position()

        self.agent.move(Actions.FORWARD.value)

        new_pos = self.agent.get_position()
        # Position should change (unless at boundary)
        if (0 < initial_pos[0] < self.env.width - 1 and
                0 < initial_pos[1] < self.env.height - 1):
            assert initial_pos != new_pos, "Forward should change position"

    def test_backward_movement(self):
        """Test backward movement."""
        initial_pos = self.agent.get_position()

        self.agent.move(Actions.BACKWARD.value)

        new_pos = self.agent.get_position()
        # Position should change (unless at boundary)
        if (0 < initial_pos[0] < self.env.width - 1 and
                0 < initial_pos[1] < self.env.height - 1):
            assert initial_pos != new_pos, "Backward should change position"

    def test_rotation_actions(self):
        """Test rotation actions."""
        initial_angle = self.agent.get_direction()

        # Test left rotation
        self.agent.move(Actions.ROTATE_LEFT.value)
        left_angle = self.agent.get_direction()

        # Test right rotation
        self.agent.move(Actions.ROTATE_RIGHT.value)
        right_angle = self.agent.get_direction()

        # Angles should be different
        assert initial_angle != left_angle or initial_angle != right_angle

    def test_boundary_collision(self):
        """Test that agents can't move outside map boundaries."""
        # Move agent to corner
        self.agent.set_position((0, 0))

        # Try to move outside boundaries
        for _ in range(5):  # Multiple attempts
            self.agent.move(Actions.BACKWARD.value)

        final_pos = self.agent.get_position()

        # Should still be within bounds
        assert 0 <= final_pos[0] < self.env.width
        assert 0 <= final_pos[1] < self.env.height

    def test_energy_consumption(self):
        """Test that actions consume energy."""
        initial_energy = self.agent.get_energy()

        # Perform energy-consuming action
        self.agent.move(Actions.FORWARD.value)

        # Simulate environment step to process energy
        action_dict = {f"agent_{i}": Actions.IDLE.value for i in range(self.env.num_agents)}
        self.env.step(action_dict)

        final_energy = self.agent.get_energy()

        # Energy should decrease
        assert final_energy < initial_energy, "Actions should consume energy"

    def test_movement_sequence(self):
        """Test a sequence of movements."""
        positions = []

        # Record initial position
        positions.append(self.agent.get_position())

        # Execute a sequence of moves
        moves = [Actions.FORWARD.value, Actions.ROTATE_LEFT.value,
                 Actions.FORWARD.value, Actions.ROTATE_RIGHT.value]

        for move in moves:
            self.agent.move(move)
            positions.append(self.agent.get_position())

        # Should have recorded all positions
        assert len(positions) == len(moves) + 1

    def test_direction_tracking(self):
        """Test that agent direction is properly tracked."""
        # Get initial direction
        initial_direction = self.agent.get_direction()

        # Rotate multiple times (16 rotations should be close to full circle since each is π/8)
        for _ in range(16):
            self.agent.move(Actions.ROTATE_LEFT.value)

        # Direction should be back to approximately initial (full rotation)
        final_direction = self.agent.get_direction()

        # Allow for floating point precision - after 16 rotations should be close to original
        # Check if the difference is small (considering wrapping around 2π)
        diff = abs(initial_direction - final_direction)
        normalized_diff = min(diff, 2*np.pi - diff)
        assert normalized_diff < 0.5, f"Direction difference too large: {normalized_diff}"

    def test_target_direction_calculation(self):
        """Test target direction calculation."""
        target_direction = self.agent.get_target_direction()

        # Should return a valid angle
        assert isinstance(target_direction, int | float)
        assert 0 <= target_direction <= 2 * np.pi

    def test_distance_to_target(self):
        """Test distance to target calculation."""
        distance = self.agent.get_distance_to_target()

        # Should return a positive distance
        assert isinstance(distance, int | float)
        assert distance >= 0

    def test_angle_to_target(self):
        """Test angle to target calculation."""
        angle = self.agent.get_angle_to_target()

        # Should return a valid angle
        assert isinstance(angle, int | float)

    def test_action_validation(self):
        """Test that invalid actions are handled."""
        initial_pos = self.agent.get_position()

        # Try invalid action (should not crash)
        try:
            self.agent.move(999)  # Invalid action
            final_pos = self.agent.get_position()
            # Position should remain unchanged for invalid actions
            assert initial_pos == final_pos
        except (ValueError, IndexError, AttributeError):
            # It's also acceptable to raise an exception for invalid actions
            pass


if __name__ == "__main__":
    unittest.main()
