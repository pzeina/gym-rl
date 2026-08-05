"""Unit tests for agent communication system."""

import unittest

from mili_env.envs.classes.robot_base import AgentRole, CommunicationMessage, MessageType
from mili_env.envs.terrain_world import TerrainWorldEnv


class TestAgentCommunication(unittest.TestCase):
    """Test cases for agent communication functionality."""

    def setUp(self):
        """Set up test environment with multiple agents."""
        self.env = TerrainWorldEnv(num_agents=3, render_mode=None)
        self.agent1 = self.env.agents[0]
        self.agent2 = self.env.agents[1]
        self.agent3 = self.env.agents[2]

    def test_message_creation(self):
        """Test creating communication messages."""
        message = CommunicationMessage(
            sender_id=0,
            receiver_id=1,
            message_type=MessageType.STATUS_UPDATE,
            content={
                "position": (10, 15),
                "health": 100,
                "energy": 90,
                "ammunition": 50,
                "team": 0,
                "role": "CAP",
            },
            timestamp=0.0
        )

        assert message.sender_id == 0
        assert message.receiver_id == 1
        assert message.message_type == MessageType.STATUS_UPDATE
        assert message.content["health"] == 100

    def test_message_sending(self):
        """Test sending messages between agents."""
        # Place agents close together
        self.agent1.set_position((10, 10))
        self.agent2.set_position((12, 10))

        # Send a status message
        self.agent1.broadcast_status()

        # Check that message was added to outbox
        assert len(self.agent1.message_outbox) > 0
        assert self.agent1.message_outbox[0].message_type == MessageType.STATUS_UPDATE

    def test_message_receiving(self):
        """Test receiving and processing messages."""
        message = CommunicationMessage(
            sender_id=1,
            receiver_id=0,
            message_type=MessageType.STATUS_UPDATE,
            content={
                "position": (5, 10),
                "health": 80,
                "energy": 70,
                "ammunition": 30,
                "team": 0,
                "role": "CAP",
            },
            timestamp=0.0
        )

        # Send message to agent
        self.agent1.receive_message(message)

        # Check message was received
        assert len(self.agent1.message_inbox) > 0
        assert self.agent1.message_inbox[0].sender_id == 1

    def test_communication_range(self):
        """Test that communication respects range limits."""
        # Place agents far apart
        self.agent1.set_position((0, 0))
        self.agent2.set_position((50, 50))  # Beyond communication range

        # Test communication range directly
        can_communicate = self.agent1.communicate(self.agent2)
        assert not can_communicate, "Agents should not be able to communicate at this distance"

        # Agents should not receive messages due to distance
        assert hasattr(self.agent2, "communication_range")
        assert hasattr(self.agent2, "message_inbox")

    def test_agent_roles(self):
        """Test agent role hierarchy."""
        # Test role enumeration exists (French hierarchy)
        assert hasattr(AgentRole, "CDU")
        assert hasattr(AgentRole, "ADU")
        assert hasattr(AgentRole, "CDS")
        assert hasattr(AgentRole, "SOA")
        assert hasattr(AgentRole, "CDG")
        assert hasattr(AgentRole, "CAP")

    def test_message_types(self):
        """Test supported message types (orders removed)."""
        message_types = [
            MessageType.STATUS_UPDATE,
            MessageType.ALLY_SPOTTED,
            MessageType.ENEMY_SPOTTED,
            MessageType.HELP_REQUEST,
        ]

        for msg_type in message_types:
            assert isinstance(msg_type.value, str)

    def test_help_request(self):
        """Test help request functionality."""
        # Set up agent in need of help
        self.agent1.set_health(30)  # Set low health

        # Request help
        help_message = self.agent1.request_help("medical")

        # Check message was created and has correct properties
        assert help_message.message_type == MessageType.HELP_REQUEST
        assert help_message.sender_id == self.agent1.agent_id
        assert len(self.agent1.message_outbox) > 0

        # Find the help request in outbox
        help_messages = [msg for msg in self.agent1.message_outbox if msg.message_type == MessageType.HELP_REQUEST]
        assert len(help_messages) > 0

    def test_enemy_reporting(self):
        """Test enemy position reporting."""
        enemy_position = (25, 30)
        enemy_message = self.agent1.report_enemy_spotted(-1, enemy_position)  # -1 for unknown enemy ID

        # Check enemy report was created and has correct properties
        assert enemy_message.message_type == MessageType.ENEMY_SPOTTED
        assert enemy_message.sender_id == self.agent1.agent_id
        assert len(self.agent1.message_outbox) > 0

        # Find the enemy report in outbox
        enemy_reports = [msg for msg in self.agent1.message_outbox if msg.message_type == MessageType.ENEMY_SPOTTED]
        assert len(enemy_reports) > 0

    def test_info_only_comms(self):
        """Test that only info messages are used and delivered within range."""
        # Place agents within range
        self.agent1.set_position((10, 10))
        self.agent2.set_position((12, 10))
        # Agent1 broadcasts status and requests help
        status_msg = self.agent1.broadcast_status()
        help_msg = self.agent1.request_help("general")
        # Outbox should contain messages
        assert status_msg.message_type == MessageType.STATUS_UPDATE
        assert help_msg.message_type == MessageType.HELP_REQUEST
        assert len(self.agent1.message_outbox) >= 2
        # Deliver messages via an environment step (processes communications)
        # A simple IDLE action for each agent is sufficient to process the outbox
        idle_actions = {
            f"agent_{i}": {
                "elementary_act": 0,
                "combat": {"move": 0, "fire_enemy": self.env.num_agents},
                "comm": {"types": [0, 0, 0], "targets": [self.env.num_agents] * 3},
                "c2": {"give_order": 0, "orders": [self.env.MISSION_NO_CHANGE_SENTINEL] * self.env.num_agents}
            } for i in range(self.env.num_agents)
        }
        self.env.step(idle_actions)
        # Agent2's knowledge should be updated about agent1
        known = self.agent2.get_known_agents()
        assert isinstance(known, dict)
        assert self.agent1.agent_id in known

    def test_message_processing(self):
        """Test message processing functionality."""
        # Create a test message
        message = CommunicationMessage(
            sender_id=1,
            receiver_id=0,
            message_type=MessageType.STATUS_UPDATE,
            content={
                "position": (15, 20),
                "health": 90,
                "energy": 85,
                "ammunition": 40,
                "team": 0,
                "role": "CAP",
            },
            timestamp=0.0
        )

        # Add to inbox
        self.agent1.receive_message(message)

        # Process messages
        self.agent1.process_messages(1.0)

        # Check that processing occurred (inbox should be cleared after processing)
        assert hasattr(self.agent1, "message_inbox")
        # After processing, inbox should be empty (messages are cleared)
        assert len(self.agent1.message_inbox) == 0

    def test_communication_statistics(self):
        """Test communication statistics tracking."""
        stats = self.agent1.get_communication_stats()

        assert isinstance(stats, dict)
        assert "messages_sent" in stats
        assert "active_orders" in stats


if __name__ == "__main__":
    unittest.main()
