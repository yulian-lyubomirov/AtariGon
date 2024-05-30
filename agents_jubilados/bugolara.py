import random
from typing import Optional
import numpy as np
from atarigon.api import Goshi, Goban, Ten
import copy
from atarigon.exceptions import KūtenError

class Bugolara(Goshi):
    def __init__(self):
        super().__init__('Bugolara')
    
    def check_liberties_and_valid_moves(self, goban) -> dict:
        liberties = {}
        for row in range(goban.size):
            for col in range(goban.size):
                if goban.ban[row][col] is None:
                    ten = Ten(row, col)
                    liberty_count = sum(
                        1 for neighbor in goban.shūi(ten) if goban.ban[neighbor.row][neighbor.col] is None
                    )
                    liberties[ten] = liberty_count
        return liberties
    

    def compute_reward(self, goban: Goban, ten: Ten) -> int:
        copy_goban = copy.deepcopy(goban)
        reward = 0
        captured = copy_goban.place_stone(ten, self)
        
        if captured:
            reward += 10 * len(captured)
        
        # Check the liberties of the player's group after placing the stone
        new_group_liberties = copy_goban.kokyū_ten(ten)

        # If player´s stone leaves only 1 liberty for a group, give negative reward
        if len(new_group_liberties) ==1:
            reward -=5
        elif len(new_group_liberties)>=1:
            reward += len(new_group_liberties)

        
        adjacent_positions = goban.shūi(ten)
        # NO DEBERIA DE DAR ERROR pero da :(
        for pos in adjacent_positions:
            if goban.ban[pos.row][pos.col] is not None and goban.ban[pos.row][pos.col] != self:
                try:
                    enemy_liberties = self.liberties(copy_goban, pos)
                    if len(enemy_liberties) == 1:
                        reward += 3
                except KūtenError:
                    # Skip empty positions
                    continue
        
        return reward
    

    # def compute_reward(self, goban: Goban, ten: Ten):
    #     copy_goban = copy.deepcopy(goban)
    #     reward = 0
    #     captured = copy_goban.place_stone(ten, self)
        
    #     if captured:
    #         reward += 5 * len(captured)
        
    #     adjacent_positions = goban.shūi(ten)
        
    #     adjacent_count = sum(
    #         1 for pos in adjacent_positions
    #         if goban.ban[pos.row][pos.col] == self
    #     )
    #     reward += adjacent_count
        
    #     is_corner = (ten.row in {0, goban.size-1}) and (ten.col in {0, goban.size-1})
        
    #     if is_corner:
    #         no_non_self_adjacent_stones = all(
    #             goban.ban[pos.row][pos.col] == self or goban.ban[pos.row][pos.col] is None
    #             for pos in adjacent_positions
    #         )
    #         if no_non_self_adjacent_stones:
    #             reward += 10
        
    #     for pos in adjacent_positions:
    #         if goban.ban[pos.row][pos.col] is not None and goban.ban[pos.row][pos.col] != self:
    #             enemy_liberties = self.liberties(goban, pos)
    #             if len(enemy_liberties) == 1:
    #                 reward += 10
        
    #     for pos in adjacent_positions:
    #         if goban.ban[pos.row][pos.col] == self:
    #             pre_liberties = self.liberties(goban, pos)
    #             if len(pre_liberties) == 1:
    #                 post_goban = copy.deepcopy(goban)
    #                 post_goban.place_stone(ten, self)
    #                 post_liberties = self.liberties(post_goban, pos)
    #                 if len(post_liberties) > 1:
    #                     reward += 15

    #     return reward
    
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
        valid_moves = self.check_liberties_and_valid_moves(goban)
        move_rewards = {}

        for move, liberties in valid_moves.items():
            reward = liberties + self.compute_reward(goban, move)
            move_rewards[move] = reward
            if reward > max_reward:
                max_reward = reward
        
        best_moves = [move for move, reward in move_rewards.items() if reward == max_reward]
        move =  random.choice(best_moves)
        # print(move)
        return move
