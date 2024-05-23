from atarigon.api import Goshi, Goban, Ten
from typing import Optional
import copy
import random

class MonteCarlo2(Goshi):
    def __init__(self, num_iterations=5):
        super().__init__("MonteCarlo2")
        self.num_iterations = num_iterations
        self.first_stone = False
        self.num_opponents = 0
        self.first_to_play = False
        self.valid_moves = []

    def get_state(self, goban):
        """Get a hashable representation of the board state."""
        return tuple(tuple(row) for row in goban.ban)

    def get_occupied_moves(self, goban):
        occupied = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is not None
        ]
        return occupied

    def get_random_move(self) -> Optional[Ten]:
        """Gets a random move from the list of valid moves."""
        if not self.valid_moves:
            return None
        return random.choice(self.valid_moves)

    def get_valid_moves(self, goban):
        """Initializes and updates the list of valid moves."""
        self.valid_moves = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]

    def remove_valid_move(self, move: Ten):
        """Removes a move from the list of valid moves."""
        if move in self.valid_moves:
            self.valid_moves.remove(move)

    def check_liberties_in_empty_positions(self, goban) -> dict:
        """Verifica las libertades en todas las posiciones vacías del tablero."""
        liberties = {}
        for ten in self.valid_moves:
            liberty_count = sum(
                1 for neighbor in goban.shūi(ten) if goban.ban[neighbor.row][neighbor.col] is None
            )
            liberties[ten] = liberty_count
        return liberties

    def decide(self, goban):
        if not self.valid_moves:
            self.get_valid_moves(goban)

        if self.num_opponents == 0 and not self.first_stone:
            self.num_opponents = len(self.get_occupied_moves(goban))
            self.first_to_play = True
            move = self.get_random_move()
            self.remove_valid_move(move)
            return move
        
        if not self.first_stone and self.first_to_play:
            self.first_stone = True
            self.num_opponents = len(self.get_occupied_moves(goban)) - 1
        
        elif not self.first_stone and not self.first_to_play and self.num_opponents == 0:
            self.num_opponents = len(self.get_occupied_moves(goban))
            _, move = self.decide_simulation(goban)
            self.remove_valid_move(move)
            return move
        
        elif not self.first_stone and not self.first_to_play and self.num_opponents > 0:
            self.num_opponents = len(self.get_occupied_moves(goban)) - self.num_opponents - 1
            self.first_stone = True

        best_move = None
        best_value = -float('inf')

        for _ in range(self.num_iterations):
            copy_goban = copy.deepcopy(goban)
            value, move = self.simulate_random_games(copy.deepcopy(copy_goban), 5)
            if value > best_value:
                best_value = value
                best_move = move

        # self.remove_valid_move(best_move)
        return best_move

    def compute_reward(self, goban, ten,captured):
        if len(captured) > 0:
            return 2 * len(captured)  # Recompensa por capturas
        if self.is_last_move(goban):
            return 10  # Gran recompensa por ser el último en poner piedra
        liberties = sum(
            1 for neighbor in goban.shūi(ten) if goban.ban[neighbor.row][neighbor.col] is None
        )
        return liberties

    def is_last_move(self, goban):
        for row in goban.ban:
            for col in row:
                if col is None:
                    return False
        return True

    def decide_simulation(self, goban):
        max_reward = -1000
        moves_dict={}
        # goban_copy=copy.deepcopy(goban)
        valid_moves = copy.deepcopy(self.valid_moves)
        for move in valid_moves:
            captured = goban.place_stone(move,self)
            if move in valid_moves:
                valid_moves.remove(move)
            reward = self.compute_reward(goban, move,captured)
            if reward > max_reward:
                max_reward = reward
                moves_dict[move]=reward
        sorted_moves_dict = sorted(moves_dict.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_moves_dict) >= 3:
            top_3_moves = sorted_moves_dict[:3]
        else:
            top_3_moves = sorted_moves_dict
        chosen_move,chosen_reward = random.choice(list(top_3_moves))
        return chosen_move,chosen_reward

    def simulate_random_games(self, goban, num_turns):
        self.get_valid_moves(goban)

        # first_move = self.get_random_move()
        first_move,total_reward = self.decide_simulation(copy.deepcopy(goban))
        self.remove_valid_move(first_move)
        captured=goban.place_stone(first_move, self)
        # total_reward = self.compute_reward(goban, first_move,captured)
        
        for _ in range(num_turns):
            if self.valid_moves:
                for _ in range(self.num_opponents):
                    random_move = self.get_random_move()
                    if random_move:
                        goban.place_stone(random_move, self)
                        self.remove_valid_move(random_move)
                    if not self.valid_moves:
                        break
            if not self.valid_moves:
                break
            move,reward = self.decide_simulation(copy.deepcopy(goban))
            if move:
                captured = goban.place_stone(move, self)
                self.remove_valid_move(move)
                total_reward += reward

        return total_reward, first_move