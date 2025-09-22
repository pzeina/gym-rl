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
                "health": 100,
                "energy": 90,
                "position": (10, 15),
                "ammunition": 50
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
                "health": 80,
                "energy": 70,
                "position": (5, 10),
                "ammunition": 30
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
        # Test role enumeration exists
        assert hasattr(AgentRole, "COMMANDER")
        assert hasattr(AgentRole, "LIEUTENANT")
        assert hasattr(AgentRole, "SOLDIER")
        assert hasattr(AgentRole, "SCOUT")
        assert hasattr(AgentRole, "MEDIC")

    def test_message_types(self):
        """Test all message types are available."""
        message_types = [
            MessageType.STATUS_UPDATE,
            MessageType.ALLY_SPOTTED,
            MessageType.ENEMY_SPOTTED,
            MessageType.ORDER_MOVE_TO,
            MessageType.ORDER_FOLLOW,
            MessageType.ORDER_ATTACK,
            MessageType.HELP_REQUEST,
            MessageType.FORMATION_KEEP,
            MessageType.RETREAT,
            MessageType.REGROUP
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

    def test_order_giving(self):
        """Test giving movement orders."""
        target_position = (20, 25)

        move_order = self.agent1.give_move_order(1, target_position)

        # Check order was created and has correct properties
        assert move_order.message_type == MessageType.ORDER_MOVE_TO
        assert move_order.sender_id == self.agent1.agent_id
        assert move_order.receiver_id == 1
        assert len(self.agent1.message_outbox) > 0

        # Find the order in outbox
        orders = [msg for msg in self.agent1.message_outbox if msg.message_type == MessageType.ORDER_MOVE_TO]
        assert len(orders) > 0

    def test_message_processing(self):
        """Test message processing functionality."""
        # Create a test message
        message = CommunicationMessage(
            sender_id=1,
            receiver_id=0,
            message_type=MessageType.STATUS_UPDATE,
            content={
                "health": 90,
                "energy": 85,
                "position": (15, 20),
                "ammunition": 40
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
