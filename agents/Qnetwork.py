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


# Define the neural network model
class QNetwork(nn.Module):
    def __init__(self,board_size):
        super(QNetwork, self).__init__()
        self.board_size = board_size
        self.fc1 = nn.Linear(board_size * board_size * 3, 256)  # Input size increased
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, board_size * board_size)

    def forward(self, x):
        x = x.view(-1, self.board_size * self.board_size * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Implementing the Q-learning agent
class Torchman(Goshi):
    def __init__(self):
        super().__init__("Torchman")
        self.board_size = 9
        self.model = QNetwork(self.board_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.2  # Exploration rate
        self.gamma = 0.9  # Discount factor
        self.state_history = []  # Store the history of states
        self.action_history = []  # Store the history of actions
        self.reward_history = []  # Store the history of rewards


    def is_game_ended(self,goban: Goban) -> bool:
        for row in goban.ban:
            for stone in row:
                if stone == self:
                    return False
        return True
    

    def compute_reward(self, goban: Goban, ten: Ten) -> int:
        copy_goban = copy.deepcopy(goban)
        reward = 0
        captured = copy_goban.place_stone(ten, self)
        
        if captured:
            reward += 10 * len(captured)
        
        # Check the liberties of the player's group after placing the stone
        new_group_liberties = copy_goban.kokyū_ten(ten)

        # If player´s stone leaves only 1 liberty for a group, give negative reward
        if len(new_group_liberties) ==1:
            reward -=10
        elif len(new_group_liberties)>=2:
            reward += len(new_group_liberties)

        
        adjacent_positions = goban.shūi(ten)
    
        for pos in adjacent_positions:
            if goban.ban[pos.row][pos.col] is not None and goban.ban[pos.row][pos.col] != self:
                try:
                    enemy_liberties = self.liberties(copy_goban, pos)
                    if len(enemy_liberties) == 1:
                        reward += 3
                except KūtenError:
                    # Skip empty positions
                    continue
        
        return reward
    
    def liberties(self, goban: Goban, pos: Ten) -> set[Ten]:
        group = self.get_group(goban, pos, set())
        liberties = set()
        for stone in group:
            neighbors = goban.shūi(stone)
            for neighbor in neighbors:
                if goban.ban[neighbor.row][neighbor.col] is None:
                    liberties.add(neighbor)
        return liberties
    
    def get_group(self, goban: Goban, pos: Ten, visited: set[Ten]) -> set[Ten]:
        group = set()
        to_visit = [pos]
        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                group.add(current)
                neighbors = [n for n in goban.shūi(current) if goban.ban[n.row][n.col] == goban.ban[pos.row][pos.col]]
                to_visit.extend(neighbors)
        return group
    
    def get_state(self, goban: 'Goban') -> np.ndarray:
        # Create a state representation with three channels:
        # 0: empty positions, 1: player's stones, -1: opponent's stones
        board = np.array(goban.ban, dtype=object)
        player_board = (board == self).astype(np.float32)
        opponent_board = ((board != self) & (board != None)).astype(np.float32)
        empty_board = (board == None).astype(np.float32)

        state = np.stack([empty_board, player_board, opponent_board], axis=-1)
        return state.flatten()

    def random_action(self,q_values: torch.Tensor, goban: 'Goban') -> int:
        valid_moves = []
        for row in range(goban.size):
            for col in range(goban.size):
                if goban.ban[row][col] is None:
                    valid_moves.append((row, col)) 

        if valid_moves:
            return random.choice(valid_moves)
        else:
            return None

    def valid_action(self, q_values: torch.Tensor, goban: 'Goban') -> int:
        q_values = q_values.squeeze().cpu().numpy()
        sorted_actions = np.argsort(-q_values)  # Sort actions by Q-value in descending order
        # print(sorted_actions)

        for action in sorted_actions:
            x, y = divmod(action, self.board_size)
            if goban.ban[x][y] == None:
                return action
        return None #sorted_actions[0]  # If no valid action is found, return the best available
    
    def decide(self, goban: 'Goban') -> Optional[Ten]:
        state = self.get_state(goban)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if self.is_game_ended(goban):
            self.train()
        if random.uniform(0, 1) < self.epsilon:
            q_values = self.model(state_tensor) #sacar?
            action = self.random_action(q_values,goban)
            x,y=action
            action = x*y
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor) #sacar?
                action = self.valid_action(q_values, goban)
                x, y = divmod(action, self.board_size)

        reward = self.compute_reward(goban,Ten(x,y))

        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
        return Ten(x, y)
    

    def train(self):
        # Compute the reward for each action and perform backpropagation
        for i in range(len(self.action_history)):
            state_tensor = torch.tensor(self.state_history[i], dtype=torch.float32).unsqueeze(0)
            action = self.action_history[i]
            reward = self.reward_history[i]
            action_tensor = torch.tensor([[action]], dtype=torch.int64)  # Ensure the correct shape

            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            # Compute the Q value for the current state and action
            q_values = self.model(state_tensor)
            # q_value = q_values.gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)
            print(f"{q_values.shape} {action_tensor} ")

            q_value = q_values.gather(1, action_tensor).squeeze(-1)

            # Compute the target Q value
            with torch.no_grad():
                target_q_value = reward_tensor

            # Compute the loss
            loss = self.criterion(q_value, target_q_value)

            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print('------------------')
        # Clear the history after training
        self.state_history = []
        self.action_history = []