import numpy as np
import random
from collections import defaultdict
from atarigon.api import Goshi, Goban,Ten
from typing import Optional
import copy

class QLearningAgent2(Goshi):
    def __init__(self,alpha=0.01, gamma=0.99, epsilon=0.1):
        super().__init__("QLearningAgent2")
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(float)  # Q-table

    def get_state(self, goban):
        """Get a hashable representation of the board state."""
        return tuple(tuple(row) for row in goban.ban)

    def decide(self, goban: 'Goban') -> Optional[Ten]:
        state = self.get_state(goban)
        if random.uniform(0, 1) < self.epsilon:
            # Explorar: movimiento aleatorio
            return self.random_move(goban)
        else:
            # Explotar: movimiento con el mayor valor Q
            return self.best_move(goban, state)

    def random_move(self, goban):
        valid_moves=self.get_valid_moves(goban)
        return random.choice(valid_moves) if valid_moves else None
    def get_valid_moves(self,goban):        
        valid_moves = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]
        return valid_moves
    
    def decide(self, goban):
        max_q_value = -float('inf')
        max_reward = -1000
        best_move = None
        valid_moves=self.get_valid_moves(goban)
        for row in range(goban.size):
            for col in range(goban.size):
                move = Ten(row, col)
                if move in (valid_moves):
                    reward = self.compute_reward(goban,move)
                    if reward > max_reward:
                        max_reward = reward
                        best_move = move
        return best_move

    def compute_reward(self, goban, ten):
        copy_goban = copy.deepcopy(goban)
        captured = copy_goban.place_stone(ten, self)
        # if copy_goban.jishi(ten, self):
        #     return -10  # Penalización fuerte por suicidio
        if len(captured) > 0:
            return 5 * len(captured)  # Recompensa por capturas
        if self.is_last_move(goban):
            return 10  # Gran recompensa por ser el último en poner piedra
        # Recompensa basada en el número de libertades
        # liberties = len(goban.kokyū_ten(ten))
        return 1

    def is_last_move(self, goban):
        for row in goban.ban:
            for col in row:
                if col is None:
                    return False
        return True

    def update_q_value(self, goban, action, reward, next_goban):
        state = self.get_state(goban)
        next_state = self.get_state(next_goban)
        next_max = max(self.q_table[(next_state, Ten(row, col))]
                       for row in range(goban.size)
                       for col in range(goban.size)
                       if goban.seichō(Ten(row, col), self))
        self.q_table[(state, action)] += self.alpha * (reward + self.gamma * next_max - self.q_table[(state, action)])

    def learn(self, goban, action, reward, next_goban):
        self.update_q_value(goban, action, reward, next_goban)

    def play_turn(self, goban):
        copy_goban = copy.deepcopy(goban)
        state = self.get_state(copy_goban)
        action = self.decide(copy_goban)
        if action is not None:
            cloned_goban = goban.clone()
            captured = goban.place_stone(action, self)
            reward = self.compute_reward(goban, action, captured)
            self.learn(cloned_goban, action, reward, goban)
        return action