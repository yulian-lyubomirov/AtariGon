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
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, board_size * board_size)

    def forward(self, x):
        x = x.view(-1, self.board_size * self.board_size * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Implementing the Q-learning agent
class QAgent(Goshi):
    def __init__(self):
        super().__init__("QAgent")
        self.board_size = 9
        self.model = QNetwork(self.board_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.9  # Discount factor
        self.action = None
        self.state = None
        self.memory = []


    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Weights saved to {path}")

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Set the model to evaluation mode
        print(f"Weights loaded from {path}")

    def is_game_ended(self,goban: Goban) -> bool:
        for row in goban.ban:
            for stone in row:
                if stone == self:
                    return False
        return True
    

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
    def get_valid_positions_from_state(self,state: np.ndarray) -> list[int]:
        # Reshape the state to its original 3-channel form
        state = state.reshape((self.board_size, self.board_size, 3))
        
        # Extract the empty positions channel (first channel)
        empty_positions = state[:, :, 0]
        
        # Identify valid actions (positions with value 1 in the empty positions channel)
        valid_positions = np.where(empty_positions == 1)
        valid_actions = [x * self.board_size + y for x, y in zip(valid_positions[0], valid_positions[1])]
        
        return valid_actions
    
    def valid_action(self, q_values: torch.Tensor, goban) -> int:
        q_values = q_values.squeeze().cpu().numpy()
        sorted_actions = np.argsort(-q_values)  # Sort actions by Q-value in descending order
        # print(sorted_actions)
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
                    return action,q_values[action]
        return None #sorted_actions[0]  # If no valid action is found, return the best available
    
    def decide(self, goban: 'Goban') -> Optional[Ten]:
        state = self.get_state(goban)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if self.is_game_ended(goban):
            # print('end')
            self.train()
        if random.uniform(0, 1) < self.epsilon:
            q_values = self.model(state_tensor) #sacar?
            action = self.random_action(q_values,goban)
            x,y=action
            action = x*y
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor) #sacar?
                action,_ = self.valid_action(q_values, goban)
                x, y = divmod(action, self.board_size)

        # reward = self.compute_reward(goban,Ten(x,y))
        # self.action_history.append(action)
        # self.state_history.append(state)
        self.action = action
        self.state = state

        # self.reward_history.append(reward)
        return Ten(x, y)
    
    def update_memory(self,reward,next_state):
        # print('update')
        self.memory.append((self.action,reward,self.state,next_state))
    

    def train(self):
        # Compute the reward for each action and perform backpropagation
        # print(self.memory)
        for i in range(len(self.memory)):
            action, reward, state, next_state = self.memory[i]
            # print(f"{action}{reward}{state}{next_state}")
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor([[action]], dtype=torch.int64)  # Ensure the correct shape

            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            # Compute the Q value for the current state and action
            q_values = self.model(state_tensor)
            q_value = q_values.gather(1, action_tensor).squeeze(-1)

            # Compute the Q value for the next state
            with torch.no_grad():
                next_q_values = self.model(next_state_tensor)
                _,max_next_q_value = self.valid_action(next_q_values, state)
                # max_next_q_value = torch.max(next_q_values)

                # Compute the target Q value using the Bellman equation
                target_q_value = reward_tensor + self.gamma * max_next_q_value

            # Compute the loss
            # print(f"{q_value}  {target_q_value}")
            loss = self.criterion(q_value, target_q_value)

            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # Clear the history after training
        self.memory = []