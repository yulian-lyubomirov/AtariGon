import numpy as np
import random
from collections import defaultdict
from atarigon.api import Goshi, Goban,Ten
from typing import Optional
import copy

class Plepito(Goshi):
    def __init__(self):
        super().__init__("Plepito")
        self.places_stone=False

    def get_state(self, goban):
        """Get a hashable representation of the board state."""
        return tuple(tuple(row) for row in goban.ban)

    # def decide(self, goban: 'Goban') -> Optional[Ten]:
    #     state = self.get_state(goban)
    #     if random.uniform(0, 1) < self.epsilon:
    #         # Explorar: movimiento aleatorio
    #         return self.random_move(goban)
    #     else:
    #         # Explotar: movimiento con el mayor valor Q
    #         return self.best_move(goban, state)

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
    def check_liberties_in_empty_positions(self,goban) -> dict:
        """Verifica las libertades en todas las posiciones vacías del tablero.

        :return: Un diccionario con las posiciones vacías como claves y el número de libertades (adyacentes vacíos) como valores.
        """
        liberties = {}
        for row in range(goban.size):
            for col in range(goban.size):
                ten = Ten(row, col)
                if goban.ban[row][col] is None:
                    # Contar las libertades para la posición vacía actual
                    liberty_count = sum(
                        1 for neighbor in goban.shūi(ten) if goban.ban[neighbor.row][neighbor.col] is None
                    )
                    liberties[ten] = liberty_count
        return liberties
    def get_occupied_moves(self,goban):
        occupied = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is not None
        ]
        return occupied

    def decide(self, goban):
        max_reward = -1000
        best_move = None
        valid_moves=self.check_liberties_in_empty_positions(goban)
        for row in range(goban.size):
            for col in range(goban.size):
                move = Ten(row, col)
                if move in valid_moves:
                    reward = valid_moves[move]+self.compute_reward(goban,move)
                    if reward > max_reward:
                        max_reward = reward
                        best_move = move
        return best_move

    def compute_reward(self, goban, ten):
        copy_goban = copy.deepcopy(goban)
        occupied_moves = self.get_occupied_moves(copy_goban)
        captured = copy_goban.place_stone(ten, self)
        # if copy_goban.jishi(ten, self):
        #     return -10  # Penalización fuerte por suicidio
        if len(captured) > 0:
            return 4 * len(captured)  # Recompensa por capturas
        if self.is_last_move(goban):
            return 10  # Gran recompensa por ser el último en poner piedra
        # Recompensa basada en el número de libertades
        # liberties = len(goban.kokyū_ten(ten))
        return 0

    def is_last_move(self, goban):
        for row in goban.ban:
            for col in row:
                if col is None:
                    return False
        return True

