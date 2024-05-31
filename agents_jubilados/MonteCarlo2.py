from atarigon.api import Goshi, Goban, Ten
from typing import Optional
import copy
import random
from atarigon.exceptions import KūtenError

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
            reward -=10
        elif len(new_group_liberties)>=2:
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

    # def compute_reward(self, goban, ten,captured):
    #     if len(captured) > 0:
    #         return 5 * len(captured)  # Recompensa por capturas
    #     if self.is_last_move(goban):
    #         return 10  # Gran recompensa por ser el último en poner piedra
    #     liberties = sum(
    #         1 for neighbor in goban.shūi(ten) if goban.ban[neighbor.row][neighbor.col] is None
    #     )
    #     return liberties

    # def is_last_move(self, goban):
    #     for row in goban.ban:
    #         for col in row:
    #             if col is None:
    #                 return False
    #     return True

    def decide_simulation(self, goban):
        max_reward = -1000
        moves_dict={}
        # goban_copy=copy.deepcopy(goban)
        # valid_moves = copy.deepcopy(self.valid_moves)
        valid_moves=self.check_liberties_and_valid_moves(goban)
        for move, liberties in valid_moves.items():
            reward = liberties + self.compute_reward(goban, move)
            moves_dict[move] = reward
            if reward > max_reward:
                max_reward = reward
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