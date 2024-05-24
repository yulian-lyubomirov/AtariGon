import random
from typing import Optional
import numpy as np
from atarigon.api import Goshi, Goban, Ten
import copy


class Bugolara(Goshi):
    """

    """

    def __init__(self):
        """Initializes the player with the given name."""
        super().__init__(f'Bugolara')
        # self.my_stones= np.zeros(shape=(19,19), dtype=int)
    
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
    
    
    def random_move(self, goban):
        valid_moves=self.get_valid_moves(goban)
        return random.choice(valid_moves) if valid_moves else None
    
    # def get_valid_moves(self,goban):        
    #     valid_moves = [
    #         Ten(row, col)
    #         for row in range(len(goban.ban))
    #         for col in range(len(goban.ban[row]))
    #         if goban.ban[row][col] is None
    #     ]
    #     return valid_moves
    
    def compute_reward(self, goban: Goban, ten: Ten):
        copy_goban = copy.deepcopy(goban)
        reward = 0
        captured = copy_goban.place_stone(ten, self)

        if len(captured) > 0:
            reward += 5 * len(captured)  # Reward for captures

        # Define adjacent positions
        adjacent_positions = [
            Ten(ten.row - 1, ten.col),
            Ten(ten.row + 1, ten.col),
            Ten(ten.row, ten.col - 1),
            Ten(ten.row, ten.col + 1),
            # Ten(ten.row + 1, ten.col + 1),
            # Ten(ten.row - 1, ten.col - 1),
            # Ten(ten.row + 1, ten.col - 1),
            # Ten(ten.row - 1, ten.col + 1),
        ]

        # Sum 1 point for each adjacent stone that belongs to self
        adjacent_count = sum(
            1 for pos in adjacent_positions
            if 0 <= pos.row < goban.size and 0 <= pos.col < goban.size and goban.ban[pos.row][pos.col] == self
        )
        reward += adjacent_count

        # Check if ten is a corner position based on goban.size
        is_corner = (ten.row in {0, goban.size-1}) and (ten.col in {0, goban.size-1})
        
        if is_corner:
            no_non_self_adjacent_stones = all(
                0 <= pos.row < goban.size and 0 <= pos.col < goban.size and 
                (goban.ban[pos.row][pos.col] == self or goban.ban[pos.row][pos.col] is None)
                for pos in adjacent_positions
            )
            if no_non_self_adjacent_stones:
                reward += 10

        # Reward for blocking an enemy stone
        for pos in adjacent_positions:
            if 0 <= pos.row < goban.size and 0 <= pos.col < goban.size and goban.ban[pos.row][pos.col] is not None and goban.ban[pos.row][pos.col] != self:
                # enemy_stone = goban.ban[pos.row][pos.col]
                enemy_liberties = self.liberties(goban, pos)
                if len(enemy_liberties) == 1:
                    reward += 10  # Reward for leaving an enemy stone with one liberty
                # Reward for saving a friendly stone from being captured

        for pos in adjacent_positions:
            if 0 <= pos.row < goban.size and 0 <= pos.col < goban.size and goban.ban[pos.row][pos.col] == self:
                pre_liberties = self.liberties(goban, pos)
                if len(pre_liberties) == 1:  # Friendly stone was in danger
                    post_goban = copy.deepcopy(goban)
                    post_goban.place_stone(ten, self)
                    post_liberties = self.liberties(post_goban, pos)
                    if len(post_liberties) > 1:
                        reward += 15  # Reward for saving a friendly stone

        return reward
    
    def get_group(self, goban: Goban, pos: Ten, visited: set[Ten]) -> set[Ten]:
        group = set()
        to_visit = [pos]
        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                group.add(current)
                neighbors = [n for n in goban.shūi(current) if goban.ban[n.row][n.col] == goban.ban[pos.row][pos.col]]
                to_visit.extend(neighbors)
        return group
    def liberties(self, goban: Goban, pos: Ten) -> set[Ten]:
        group = self.get_group(goban, pos, set())
        liberties = set()
        for stone in group:
            neighbors = goban.shūi(stone)
            for neighbor in neighbors:
                if goban.ban[neighbor.row][neighbor.col] is None:
                    liberties.add(neighbor)

    
        return liberties
    
    def decide(self, goban):
        max_reward = -1000
        valid_moves = self.check_liberties_in_empty_positions(goban)
        move_rewards = {}
        
        for move in valid_moves:
            reward = valid_moves[move] + self.compute_reward(goban, move)
            move_rewards[move] = reward
            if reward > max_reward:
                max_reward = reward
        
        best_moves = [move for move, reward in move_rewards.items() if reward == max_reward]
        # best_move = random.choice(best_moves)
        # self.my_stones[best_move.row][best_move.col]=1
        return random.choice(best_moves)