import numpy as np
import random

class QLearningAgent:
    def __init__(self, action_space, observation_space, learning_rate=0.01, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__("QLearningAgent")
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = 0.01
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((observation_space.shape[0], observation_space.shape[1], action_space.n))

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            row, col = state.shape
            return np.argmax(self.q_table[row, col])

    def update_q_values(self, state, action, reward, next_state):
        row, col = state.shape
        next_row, next_col = next_state.shape
        best_next_action = np.argmax(self.q_table[next_row, next_col])
        td_target = reward + self.discount_factor * self.q_table[next_row, next_col, best_next_action]
        td_error = td_target - self.q_table[row, col, action]
        self.q_table[row, col, action] += self.learning_rate * td_error

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
