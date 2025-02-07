import matplotlib.pyplot as plt
from collections import deque

class AgentEnvInteractionVisualization:
    def __init__(self, window_width: int, window_height: int, panel_width: int, buffer_size: int = 200) -> None:
        """Initialize the visualization panel."""
        self.window_width: int = window_width
        self.window_height: int = window_height
        self.panel_width: int = panel_width
        self.buffer_size: int = buffer_size

        self.fig, self.axs = plt.subplots(3, 1, figsize=(panel_width / 100, window_height / 100))
        self.fig.tight_layout(pad=3.0)

        self.rewards = deque(maxlen=buffer_size)
        self.distances = deque(maxlen=buffer_size)
        self.action_counts = [0] * 5  # Initialize counters for 5 actions

    def update(self, reward: float, distance: float, action: int) -> None:
        """Update the visualization panel with the latest information."""
        self.rewards.append(reward)
        self.distances.append(distance)
        self.action_counts[action] += 1

        self.axs[0].cla()
        self.axs[0].plot(self.rewards, label='Reward')
        self.axs[0].legend()

        self.axs[1].cla()
        self.axs[1].plot(self.distances, label='Distance')
        self.axs[1].legend()

        self.axs[2].cla()
        self._plot_actions()

        plt.draw()
        plt.pause(0.001)

    def _plot_actions(self) -> None:
        """Plot the frequency of actions taken."""
        self.axs[2].cla()
        self.axs[2].set_xlabel('Action')
        self.axs[2].set_ylabel('Frequency')

        action_labels = ['-', '↑', '↓', '↶', '↷']
        action_colors = ['black', 'green', 'red', 'blue', 'purple']

        self.axs[2].bar(action_labels, self.action_counts, color=action_colors)

class GradientLossVisualization:
    def __init__(self, window_width: int, window_height: int, panel_width: int, buffer_size: int = 200) -> None:
        """Initialize the visualization panel."""
        self.window_width: int = window_width
        self.window_height: int = window_height
        self.panel_width: int = panel_width
        self.buffer_size: int = buffer_size

        self.fig, self.axs = plt.subplots(2, 1, figsize=(panel_width / 100, window_height / 100))
        self.fig.tight_layout(pad=3.0)

        self.losses = deque(maxlen=buffer_size)
        self.gradients = deque(maxlen=buffer_size)

    def track_loss(self, loss) -> None:
        """Track the loss of the model."""
        self.losses.append(loss)
        self._update_plot()

    def track_gradients(self, model) -> None:
        """Track the gradients of the model."""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.gradients.append(total_norm)
        self._update_plot()

    def _update_plot(self) -> None:
        """Update the visualization panel with the latest information."""
        self.axs[0].cla()
        self.axs[0].plot(self.losses, label='Loss')
        self.axs[0].legend()

        self.axs[1].cla()
        self.axs[1].plot(self.gradients, label='Gradient Magnitude')
        self.axs[1].legend()

        plt.draw()
        plt.pause(0.001)