import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Initialize a neural network with two linear layers."""
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network."""
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth') -> None:
        """Save the model to a file."""
        model_folder_path = Path(__file__).resolve().parent.parent / 'model'
        if not model_folder_path.exists():
            model_folder_path.mkdir(parents=True, exist_ok=True)

        file_path = model_folder_path / file_name
        torch.save(self.state_dict(), file_path)

    def load(self, file_name='model.pth') -> None:
        """Load the model from a file."""
        model_folder_path = Path(__file__).resolve().parent.parent / 'model'
        file_path = model_folder_path / file_name
        self.load_state_dict(torch.load(file_path))


class QTrainer:
    def __init__(self, model, lr, gamma, visualization=None) -> None:
        """Initialize the Q-learning trainer with a model, learning rate, and discount factor."""
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.visualization = visualization

    def train_step(self, state, action, reward, next_state, done) -> None:
        """Train the model using a single experience."""
        device = next(self.model.parameters()).device  # Get the device of the model
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][action[idx].item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Track gradients
        if self.visualization:
            self.visualization.track_gradients(self.model)

        self.optimizer.step()

        # Track loss
        if self.visualization:
            self.visualization.track_loss(loss.item())


