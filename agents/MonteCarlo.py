
from atarigon.api import Goshi, Goban,Ten
from typing import Optional
import copy
import random
from typing import Optional

class MonteCarlo(Goshi):
    def __init__(self,):
        super().__init__("MonteCarlo")
        self.first_stone = False # para calcular numbero de oponentes
        self.num_opponents = 0
        self.num_iterations=10

    def get_state(self, goban):
        """Get a hashable representation of the board state."""
        return tuple(tuple(row) for row in goban.ban)
    
    def get_occupied_moves(self,goban):
        occupied = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is not None
        ]
        return occupied

    def get_random_move(self, goban: 'Goban') -> Optional[Ten]:
        """Gets a random empty position in the board.

        :param goban: The current observation of the game.
        :return: The next move as a (row, col) tuple.
        """
        # Finds all the empty positions in the observation
        empty_positions = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]

        # Chooses a random valid empty position
        random.shuffle(empty_positions)
        for ten in empty_positions:
            if goban.ban[ten.row][ten.col] is None:
                return ten
        else:
            return None
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
    
    def decide(self, goban):
        # Implementar MCTS para tomar decisiones informadas
        # if len(self.get_occupied_moves(goban))==0:
        #     return self.random_move(goban)
        if self.num_opponents==0 and self.first_stone is False:
            self.num_opponents = len(self.get_occupied_moves(goban))
            self.first_to_play=True
            return self.get_random_move(goban)
        
        if self.first_stone is False and self.first_to_play is True:
            self.first_stone = True
            self.num_opponents = len(self.get_occupied_moves(goban)) -1#self.num_opponents
        
        elif self.first_stone is False and self.first_to_play is False and self.num_opponents ==0:
            self.num_opponents= len(self.get_occupied_moves(goban))
            r,move = self.decide_simulation(goban)
            return move
        elif self.first_stone is False and self.first_to_play is False and self.num_opponents>0:
            self.num_opponents = len(self.get_occupied_moves) - self.num_opponents -1
            self.first_stone =True

        for _ in range(self.num_iterations):
            best_move = None
            best_value = -float('inf')

            value, move = self.simulate_random_games(copy.deepcopy(goban),10)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move
    
    def compute_reward(self, goban, ten):
        occupied_moves = self.get_occupied_moves(goban)
        captured = goban.place_stone(ten, self)
        # if goban.jishi(ten, self):
        #     return -10  # Penalización fuerte por suicidio
        if len(captured) > 0:
            return 2 * len(captured)  # Recompensa por capturas
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

    def decide_simulation(self, goban):
        max_reward = -1000
        best_move = None
        valid_moves=self.check_liberties_in_empty_positions(goban)
        for row in range(goban.size):
            for col in range(goban.size):
                move = Ten(row, col)
                if move in (valid_moves.items()):
                    reward = valid_moves[move]+self.compute_reward(goban,move)
                    if reward > max_reward:
                        max_reward = reward
                        best_move = move
        return best_move,max_reward
    
    def simulate_random_games(self, goban,num_turns):
        first_move,reward = self.decide_simulation(goban)
        for _ in range(num_turns):
            for i in range(self.num_opponents):
                random_move = self.get_random_move(goban)
                goban.place_stone(random_move, self)
            move,r = self.decide_simulation(goban)
            goban.place_stone(move, self)
            reward += r

        return reward,move
            
            
