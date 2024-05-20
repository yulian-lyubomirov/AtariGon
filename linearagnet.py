import torch
import torch.nn as nn
import torch.optim as optim
import random
from atarigon.api import Goshi, Goban, Ten

class LinearDecisionAgent(Goshi):
    def __init__(self, state_size, action_size, epsilon=0.01, lr=0.001):
        super().__init__('LinearDecisionAgent')
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.lr = lr
        self.decision_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self.optimizer = optim.Adam(self.decision_network.parameters(), lr=self.lr)

    def decide(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            decision_scores = self.decision_network(state_tensor)
            return torch.argmax(decision_scores).item()

    def learn(self, state, action, reward, next_state):
        # This agent does not learn from individual experiences
        pass
