import numpy as np
import random
from typing import Optional, List, Dict

from atarigon.api import Goshi, Goban, Ten

class QLearningAgent(Goshi):
    def __init__(self, epsilon: float = 0.01, alpha: float = 0.01, gamma: float = 0.9):
        super().__init__('QLearningAgent')
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.q_table = {}

    def decide(self, goban: 'Goban') -> Optional[Ten]:
        empty_positions = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(empty_positions)
        else:
            q_values = [self.q_value(goban, pos) for pos in empty_positions]
            best_index = np.argmax(q_values)
            return empty_positions[best_index]

    def learn(self, state: 'Goban', action: Ten, reward: float, next_state: Optional['Goban']) -> None:
        current_q = self.q_value(state, action)
        if next_state is None or self.is_terminal(next_state):
            max_future_q = 0
        else:
            max_future_q = np.max([self.q_value(next_state, pos) for pos in next_state.empty_positions])

        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[(state, action)] = new_q

    def q_value(self, state: 'Goban', action: Ten) -> float:
        if (state, action) not in self.q_table:
            return 0
        else:
            return self.q_table[(state, action)]

    def is_terminal(self, state: 'Goban') -> bool:
        return not any(Ten(row, col) for row in range(len(state.ban)) for col in range(len(state.ban[row])) if state.ban[row][col] is None)
