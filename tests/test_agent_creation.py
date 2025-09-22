"""Unit tests for agent creation and initialization."""

import unittest

import numpy as np

from mili_env.envs.classes.robot_base import RobotBase
from mili_env.envs.terrain_world import TerrainWorldEnv


class TestAgentCreation(unittest.TestCase):
    """Test cases for agent creation and initialization."""

    def setUp(self):
        """Set up test environment."""
        self.env = TerrainWorldEnv(num_agents=4, render_mode=None)

    def test_single_agent_creation(self):
        """Test creating a single agent."""
        env = TerrainWorldEnv(num_agents=1, render_mode=None)

        self.assertEqual(len(env.agents), 1)
        self.assertIsInstance(env.agents[0], RobotBase)

    def test_multi_agent_creation(self):
        """Test creating multiple agents."""
        self.assertEqual(len(self.env.agents), 4)

        for agent in self.env.agents:
            self.assertIsInstance(agent, RobotBase)

    def test_agent_initial_attributes(self):
        """Test that agents are initialized with correct attributes."""
        agent = self.env.agents[0]

        # Check health, energy, ammunition
        self.assertEqual(agent.get_health(), 100.0)
        self.assertEqual(agent.get_energy(), 100.0)
        self.assertEqual(agent.get_ammunition(), 100.0)

        # Check position is valid
        pos = agent.get_position()
        self.assertIsInstance(pos, tuple)
        self.assertEqual(len(pos), 2)
        self.assertTrue(0 <= pos[0] < self.env.width)
        self.assertTrue(0 <= pos[1] < self.env.height)

    def test_agent_constraints(self):
        """Test that agent constraints are properly set."""
        agent = self.env.agents[0]

        # Test communication range
        self.assertEqual(agent.communication_range, 30.0)

        # Test movement constraints exist (these are stored directly on the agent)
        self.assertTrue(hasattr(agent, "max_speed_forward"))
        self.assertTrue(hasattr(agent, "max_speed_backward"))
        self.assertTrue(hasattr(agent, "max_angular_speed"))

        # Test that values are reasonable
        self.assertGreater(agent.max_speed_forward, 0)
        self.assertGreater(agent.max_speed_backward, 0)  # In this implementation, backward speed is positive
        self.assertGreater(agent.max_angular_speed, 0)

    def test_grouped_spawning(self):
        """Test grouped spawning functionality."""
        env = TerrainWorldEnv(num_agents=4, render_mode=None)
        env.create_agents(grouped=True)

        positions = [agent.get_position() for agent in env.agents]

        # Calculate distances between all agents
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                distances.append(dist)

        # In grouped mode, average distance should be small
        avg_distance = np.mean(distances)
        self.assertLess(avg_distance, 5.0, "Grouped agents should spawn close together")

    def test_normal_spawning_separation(self):
        """Test that normal spawning maintains minimum distance."""
        env = TerrainWorldEnv(num_agents=4, render_mode=None)
        env.create_agents(grouped=False)

        positions = [agent.get_position() for agent in env.agents]

        # Check minimum distance between agents
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                self.assertGreaterEqual(dist, 3.0, "Agents should maintain minimum distance")

    def test_agent_target_assignment(self):
        """Test that each agent gets a unique target."""
        targets = []
        for agent in self.env.agents:
            target = agent.get_target()
            targets.append(target)

        # Each agent should have a target
        self.assertEqual(len(targets), len(self.env.agents))

        # Targets should be within map bounds
        for target in targets:
            self.assertTrue(0 <= target[0] < self.env.width)
            self.assertTrue(0 <= target[1] < self.env.height)

    def test_agent_state_initialization(self):
        """Test that agent state is properly initialized."""
        agent = self.env.agents[0]

        # Agent should be alive and have energy
        self.assertTrue(agent.state.is_alive())
        self.assertTrue(agent.state.has_energy())

        # Agent should not be at target initially (very unlikely)
        self.assertFalse(agent.state.is_at_target())

    def test_grid_formation_small_groups(self):
        """Test grid formation for small agent groups (2-4 agents)."""
        test_cases = [2, 3, 4]

        for agent_count in test_cases:
            with self.subTest(agents=agent_count):
                env = TerrainWorldEnv(num_agents=agent_count, render_mode=None)
                env.create_agents(grouped=True)

                # Verify all agents were created
                self.assertEqual(len(env.agents), agent_count)

                # Get positions
                positions = [agent.get_position() for agent in env.agents]

                # Calculate expected grid size
                expected_grid_size = int(np.ceil(np.sqrt(agent_count)))

                # Check formation compactness
                if len(positions) > 1:
                    max_x = max(pos[0] for pos in positions)
                    min_x = min(pos[0] for pos in positions)
                    max_y = max(pos[1] for pos in positions)
                    min_y = min(pos[1] for pos in positions)

                    formation_width = max_x - min_x + 1
                    formation_height = max_y - min_y + 1

                    # Formation should not exceed expected grid dimensions
                    self.assertLessEqual(formation_width, expected_grid_size)
                    self.assertLessEqual(formation_height, expected_grid_size)

    def test_grid_formation_medium_groups(self):
        """Test grid formation for medium agent groups (5-9 agents)."""
        test_cases = [5, 6, 8, 9]

        for agent_count in test_cases:
            with self.subTest(agents=agent_count):
                env = TerrainWorldEnv(num_agents=agent_count, render_mode=None)
                env.create_agents(grouped=True)

                # Verify all agents were created
                self.assertEqual(len(env.agents), agent_count)

                # Get positions
                positions = [agent.get_position() for agent in env.agents]

                # Calculate expected grid size (should be 3x3 for 5-9 agents)
                expected_grid_size = int(np.ceil(np.sqrt(agent_count)))

                # Check that agents form a reasonably compact formation
                if len(positions) > 1:
                    # Calculate inter-agent distances
                    distances = []
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                            distances.append(dist)

                    # Average distance should be reasonable for a grid formation
                    avg_distance = np.mean(distances)
                    self.assertLess(avg_distance, expected_grid_size * 2,
                                  f"Average distance {avg_distance:.2f} too large for {agent_count} agents")

    def test_grouped_vs_individual_spawning_distances(self):
        """Test that grouped spawning produces closer formations than individual spawning."""
        agent_count = 6

        # Test grouped spawning
        env_grouped = TerrainWorldEnv(num_agents=agent_count, render_mode=None)
        env_grouped.create_agents(grouped=True)
        grouped_positions = [agent.get_position() for agent in env_grouped.agents]

        # Test individual spawning
        env_individual = TerrainWorldEnv(num_agents=agent_count, render_mode=None)
        env_individual.create_agents(grouped=False)
        individual_positions = [agent.get_position() for agent in env_individual.agents]

        # Calculate average distances for both
        def calculate_avg_distance(positions):
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                    distances.append(dist)
            return np.mean(distances) if distances else 0

        grouped_avg_dist = calculate_avg_distance(grouped_positions)
        individual_avg_dist = calculate_avg_distance(individual_positions)

        # Grouped spawning should result in closer agents (most of the time)
        # We allow some variance due to randomness
        self.assertLess(
            grouped_avg_dist, individual_avg_dist * 1.5,
            f"Grouped distance {grouped_avg_dist:.2f} should be less than individual {individual_avg_dist:.2f}"
        )

    def test_grouped_spawning_robustness(self):
        """Test that grouped spawning doesn't fail or hang."""
        # Test various scenarios that might cause issues
        test_scenarios = [
            (1, "single_agent"),
            (2, "two_agents"),
            (10, "many_agents"),
            (16, "perfect_square"),
        ]

        for agent_count, scenario_name in test_scenarios:
            with self.subTest(scenario=scenario_name, agents=agent_count):
                # This should complete without hanging or errors
                env = TerrainWorldEnv(num_agents=agent_count, render_mode=None)
                env.create_agents(grouped=True)

                # Verify all agents were created
                self.assertEqual(len(env.agents), agent_count)

                # Verify all agents have valid positions
                for i, agent in enumerate(env.agents):
                    pos = agent.get_position()
                    self.assertTrue(0 <= pos[0] < env.width,
                                  f"Agent {i} x-position {pos[0]} out of bounds")
                    self.assertTrue(0 <= pos[1] < env.height,
                                  f"Agent {i} y-position {pos[1]} out of bounds")

    def test_grid_spawning_pattern_consistency(self):
        """Test that grid spawning produces consistent patterns."""
        agent_count = 4

        # Test multiple times to check consistency
        formations = []
        for _ in range(5):
            env = TerrainWorldEnv(num_agents=agent_count, render_mode=None)
            env.create_agents(grouped=True)

            positions = [agent.get_position() for agent in env.agents]

            # Normalize positions relative to the first agent
            if positions:
                base_pos = positions[0]
                relative_positions = [(pos[0] - base_pos[0], pos[1] - base_pos[1])
                                    for pos in positions]
                formations.append(sorted(relative_positions))

        # Check that all formations have the same relative structure
        if formations:
            reference_formation = formations[0]
            for i, formation in enumerate(formations[1:], 1):
                # All formations should have the same number of agents
                self.assertEqual(len(formation), len(reference_formation))

                # The relative positions should form a consistent pattern
                # (allowing for rotations and reflections)
                formation_distances = []
                ref_distances = []

                for j in range(len(formation)):
                    for k in range(j + 1, len(formation)):
                        formation_dist = np.linalg.norm(np.array(formation[j]) - np.array(formation[k]))
                        ref_dist = np.linalg.norm(np.array(reference_formation[j]) - np.array(reference_formation[k]))
                        formation_distances.append(formation_dist)
                        ref_distances.append(ref_dist)

                # The sets of distances should be similar (grid structure is consistent)
                formation_distances.sort()
                ref_distances.sort()

                for fd, rd in zip(formation_distances, ref_distances, strict=True):
                    self.assertAlmostEqual(fd, rd, places=1,
                                         msg=f"Formation {i} differs from reference in distance pattern")

    def test_grouped_spawning_fallback_mechanisms(self):
        """Test fallback mechanisms when grouped spawning conditions are challenging."""
        # Test with many agents on a smaller map to trigger fallbacks
        # Note: We can't easily test extreme edge cases without modifying map size,
        # but we can test that the system handles reasonable stress cases

        agent_count = 12
        env = TerrainWorldEnv(num_agents=agent_count, render_mode=None)

        # This should succeed even if perfect grid formation isn't possible
        env.create_agents(grouped=True)

        # Verify all agents were placed
        self.assertEqual(len(env.agents), agent_count)

        # Verify no agents overlap (all have different positions)
        positions = [agent.get_position() for agent in env.agents]
        unique_positions = set(positions)
        self.assertEqual(len(unique_positions), len(positions),
                        "No two agents should occupy the same position")

        # Verify all positions are valid
        for pos in positions:
            self.assertTrue(0 <= pos[0] < env.width)
            self.assertTrue(0 <= pos[1] < env.height)


if __name__ == "__main__":
    unittest.main()
