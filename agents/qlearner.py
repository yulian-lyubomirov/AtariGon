import numpy as np
import random
from collections import defaultdict
from typing import Optional
import copy
from atarigon.api import Goshi, Goban, Ten
from typing import List, Optional, Set, NamedTuple

from atarigon.exceptions import (
    NotEnoughPlayersError,
    SmallBoardError,
    InvalidMoveError,
    HikūtenError, KūtenError,
)

class QLearningAgent(Goshi):
    def __init__(self, epsilon: float = 0.01, alpha: float = 0.01, gamma: float = 0.9):
        super().__init__('QLearningAgent')
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.q_table = {}


    def get_valid_moves(self, goban):
        """Initializes and updates the list of valid moves."""
        valid_moves = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]
        return valid_moves


    def decide(self, goban):
        empty_positions = self.get_valid_moves(goban)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(empty_positions)
        else:
            q_values = [self.q_value(goban, pos) for pos in empty_positions]
            best_index = np.argmax(q_values)
            move = empty_positions[best_index]
            self.learn(goban,move)
            return empty_positions[best_index]
        

    def compute_reward(self, goban, ten):
        captured = self.check_captures(ten, goban)
        if len(captured) > 0:
            return 2 * len(captured)  # Recompensa por capturas
        if self.is_terminal(goban):
            return 10  # Gran recompensa por ser el último en poner piedra
        liberties = sum(
            1 for neighbor in goban.shūi(ten) if goban.ban[neighbor.row][neighbor.col] is None
        )
        return liberties
    

    def learn(self, goban, action) -> None:
        state = goban
        next_state = goban
        current_q = self.q_value(state, action)
        reward = self.compute_reward(state, action)
        if self.is_terminal(next_state):
            max_future_q = 0
        else:
            max_future_q = np.max([self.q_value(next_state, pos) for pos in self.get_valid_moves(next_state)])

        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[(state, action)] = new_q
        self.alpha = self.alpha-0.0005

    # def learn(self, goban: Goban, action: Ten) -> None:
    #     state_key = self.hash_state(goban)
    #     print(state_key)
    #     current_q = self.q_value(goban, action)
    #     captured = goban.place_stone(action, self)
    #     reward = self.compute_reward(goban, action, captured)

    #     if self.is_terminal(goban):
    #         max_future_q = 0
    #     else:
    #         next_state_key = self.hash_state(goban)
    #         max_future_q = max(self.q_value(next_state_key, pos) for pos in self.get_valid_moves(goban))

    #     new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
    #     self.q_table[(state_key, action)] = new_q
    #     self.alpha = max(0.001, self.alpha - 0.0005)  # Avoid negative alpha

    def q_value(self, state: 'Goban', action: Ten) -> float:
        if (state, action) not in self.q_table:
            return 0
        else:
            return self.q_table[(state, action)]

    def is_terminal(self, state: 'Goban') -> bool:
        return not any(Ten(row, col) for row in range(len(state.ban)) for col in range(len(state.ban[row])) if state.ban[row][col] is None)
    def check_captures(self, ten: Ten, goban) -> Set[Goshi]:
        """Checks whether the group at the given position has liberties.

        :param ten: The position to check.
        :param goshi: The player making the move.
        :return: A set with the players that were captured. If no players
            were captured, the set is empty.
        """
        if not goban.goban_no_naka(ten):
            raise InvalidMoveError(ten)
        row, col = ten
        if goban.ban[row][col] is not None:
            raise HikūtenError(ten)
        captured = set()
        for betsu_no_ten in goban.shūi(ten):
            taisen_aite = goban.ban[betsu_no_ten.row][betsu_no_ten.col]
            if taisen_aite is None:
                # If the neighbor is empty, we don't capture anything
                continue
            if taisen_aite == self:
                # If the neighbor is us, we don't capture anything
                continue
            if goban.kokyū_ten(betsu_no_ten):
                # The neighbor has liberties, we don't capture it
                continue

            # The neighbour has no liberties, so we capture it
            goban.toru(betsu_no_ten)
            captured.add(taisen_aite)
        return captured
    