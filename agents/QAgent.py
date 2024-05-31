import abc
import random
from typing import Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from atarigon.api import Goshi, Goban, Ten
from atarigon.exceptions import KūtenError
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define the neural network model
class QNetwork(nn.Module):
    def __init__(self,board_size):
        super(QNetwork, self).__init__()
        self.board_size = board_size
        self.fc1 = nn.Linear(board_size * board_size * 3, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, board_size * board_size)

    def forward(self, x):
        x = x.view(-1, self.board_size * self.board_size * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QAgent(Goshi):
    def __init__(self):
        super().__init__("QAgent")
        self.board_size = 19
        self.model = QNetwork(self.board_size)
        self.learning_rate = 0.1
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.2  # exploration rate for training purposes
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.epsilon_decay = 0.95  # Decay rate for epsilon
        self.gamma = 0.9  # Discount factor
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=2000, eta_min=0.01) #lr scheduler
        self.action = None
        self.state = None
        self.memory = []
        self.load_weights(path='QAgent_weights2.pth')

    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)
        # print(f"Weights saved to {path}")

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Set the model to evaluation mode
        print(f"Weights loaded from {path}")

    def is_game_ended(self, goban: Goban) -> bool:
        for row in goban.ban:
            for stone in row:
                if stone == self:
                    return False
        return True

    def get_state(self, goban: 'Goban') -> np.ndarray:
        board = np.array(goban.ban, dtype=object)
        player_board = (board == self).astype(np.float32)
        opponent_board = ((board != self) & (board != None)).astype(np.float32)
        empty_board = (board == None).astype(np.float32)

        state = np.stack([empty_board, player_board, opponent_board], axis=-1)
        return state.flatten()

    def random_action(self, q_values: torch.Tensor, goban: 'Goban') -> int:
        valid_moves = []
        for row in range(goban.size):
            for col in range(goban.size):
                if goban.ban[row][col] is None:
                    valid_moves.append((row, col)) 

        if valid_moves:
            return random.choice(valid_moves)
        else:
            return None

    def get_valid_positions_from_state(self, state: np.ndarray) -> list[int]:
        state = state.reshape((self.board_size, self.board_size, 3))
        empty_positions = state[:, :, 0]
        valid_positions = np.where(empty_positions == 1)
        valid_actions = [x * self.board_size + y for x, y in zip(valid_positions[0], valid_positions[1])]
        return valid_actions

    def valid_action(self, q_values: torch.Tensor, goban) -> int:
        q_values = q_values.squeeze().cpu().numpy()
        sorted_actions = np.argsort(-q_values)
        if not isinstance(goban, Goban):
            valid_positions = self.get_valid_positions_from_state(goban)
            for action in sorted_actions:
                if action in valid_positions:
                    return action, q_values[action]
            return None
        else:
            for action in sorted_actions:
                x, y = divmod(action, self.board_size)
                if goban.ban[x][y] == None:
                    return action, q_values[action]
        return None

    def decide(self, goban: 'Goban') -> Optional[Ten]:
        state = self.get_state(goban)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # if self.is_game_ended(goban):
        #     self.train()
        if random.uniform(0, 1) < self.epsilon:
            q_values = self.model(state_tensor)
            action = self.random_action(q_values, goban)
            x, y = action
            action = x * y
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action, _ = self.valid_action(q_values, goban)
                x, y = divmod(action, self.board_size)

        self.action = action
        self.state = state
        return Ten(x, y)

    def update_memory(self, reward, next_state):
        self.memory.append((self.action, reward, self.state, next_state))

    def train(self):
        for i in range(len(self.memory)):
            action, reward, state, next_state = self.memory[i]
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor([[action]], dtype=torch.int64)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            q_values = self.model(state_tensor)
            q_value = q_values.gather(1, action_tensor).squeeze(-1)

            with torch.no_grad():
                next_q_values = self.model(next_state_tensor)
                _, max_next_q_value = self.valid_action(next_q_values, state)
                target_q_value = reward_tensor + self.gamma * max_next_q_value

            loss = self.criterion(q_value, target_q_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.save_weights("QAgent_weights2.pth")
            
        self.scheduler.step()
        self.memory = []
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # def adjust_learning_rate(self):
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] *= self.epsilon_decay