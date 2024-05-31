import random
from typing import Optional, Dict
import numpy as np
from atarigon.api import Goshi, Goban, Ten
import copy


class Bugolara(Goshi):
    def __init__(self):
        super().__init__(f'Bugolaralisto')
        self.my_stones = np.zeros(shape=(19, 19), dtype=int)
        self.q_table = {}  # Initialize the Q-table
        self.alpha = 0.01   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.02  # Exploration rate

    def check_liberties_in_empty_positions(self, goban) -> Dict[Ten, int]:
        liberties = {}
        for row in range(goban.size):
            for col in range(goban.size):
                ten = Ten(row, col)
                if goban.ban[row][col] is None:
                    liberty_count = sum(
                        1 for neighbor in goban.shūi(ten) if goban.ban[neighbor.row][neighbor.col] is None
                    )
                    liberties[ten] = liberty_count
        return liberties

    def random_move(self, goban):
        valid_moves = self.get_valid_moves(goban)
        return random.choice(valid_moves) if valid_moves else None

    def compute_reward(self, goban: Goban, ten: Ten) -> int:
        copy_goban = copy.deepcopy(goban)
        reward = 0
        captured = copy_goban.place_stone(ten, self)

        if len(captured) > 0:
            reward += 5 * len(captured)  # Reward for captures

        adjacent_positions = [
            Ten(ten.row - 1, ten.col),
            Ten(ten.row + 1, ten.col),
            Ten(ten.row, ten.col - 1),
            Ten(ten.row, ten.col + 1),
            Ten(ten.row + 1, ten.col + 1),
            Ten(ten.row - 1, ten.col - 1),
            Ten(ten.row + 1, ten.col - 1),
            Ten(ten.row - 1, ten.col + 1),
        ]

        adjacent_count = sum(
            1 for pos in adjacent_positions
            if 0 <= pos.row < goban.size and 0 <= pos.col < goban.size and self.my_stones[pos.row, pos.col] == 1
        )
        reward += adjacent_count

        is_corner = (ten.row in {0, goban.size-1}) and (ten.col in {0, goban.size-1})
        if is_corner:
            no_non_self_adjacent_stones = all(
                0 <= pos.row < goban.size and 0 <= pos.col < goban.size and 
                (self.my_stones[pos.row, pos.col] == 1 or copy_goban.ban[pos.row][pos.col] is None)
                for pos in adjacent_positions
            )
            if no_non_self_adjacent_stones:
                reward += 20

        return reward

    def get_valid_moves(self, goban: Goban):
        valid_moves = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]
        return valid_moves

    def update_q_table(self, current_state: str, action: Ten, reward: int, next_goban: Goban):
        max_next_q = max(
            self.q_table.get((str(next_goban.ban), a), 0.0)
            for a in self.get_valid_moves(next_goban)
        )
        current_q = self.q_table.get((current_state, action), 0.0)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(current_state, action)] = new_q

    def choose_action(self, goban: Goban):
        if random.uniform(0, 1) < self.epsilon:
            return self.random_move(goban)
        else:
            state = str(goban.ban)
            valid_moves = self.get_valid_moves(goban)
            q_values = {move: self.q_table.get((state, move), 0.0) for move in valid_moves}
            max_q = max(q_values.values())
            best_moves = [move for move, q in q_values.items() if q == max_q]
            return random.choice(best_moves) if best_moves else None

    def decide(self, goban: Goban):
        current_state = str(goban.ban)
        action = self.choose_action(goban)
        if action:
            reward = self.compute_reward(goban, action)
            new_goban = copy.deepcopy(goban)
            new_goban.place_stone(action, self)
            self.update_q_table(current_state, action, reward, new_goban)
        return action
