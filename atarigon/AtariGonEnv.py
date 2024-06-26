import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from api import Goban, Ten, Goshi, NotEnoughPlayersError, SmallBoardError, InvalidMoveError, HikūtenError, KūtenError

class AtariGonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,players, player,size=19):
        super(AtariGonEnv, self).__init__()
        self.size = size
        self.players = players
        self.player = player
        self.reset()

    # def update_board(self,ten,player):
    #     self.goban.place_stone(ten,player)
    def is_game_ended(self) -> bool:
        for row in self.goban.ban:
            for stone in row:
                if stone == self.player:
                    return False
        return True
    def step(self, ten, player):
    
        if player.name == self.player.name:
            captured = self.goban.place_stone(ten, self.player)
            done = self.is_game_ended()
            reward = self.compute_reward(ten,captured)
            self.turn += 1
            # print('-----------')
            # print('yo')
            # self.render()

            obs = self.get_state()
            return obs, reward, done, {}
        else:
            captured = self.goban.place_stone(ten,player)
            # print('-----------')
            # print('el')
            # self.render()
            done = self.is_game_ended()

        return None,None,done,None

    def reset(self):
        self.goban = Goban(size=self.size, goshi=self.players)
        self.turn = 0
        self.max_turns = self.size * self.size
        return self.goban

    def render(self, mode='human'):
        self.goban.print_board()

    def _get_obs(self):
        board = np.zeros((self.size, self.size), dtype=np.int)
        for row in range(self.size):
            for col in range(self.size):
                if self.goban.ban[row][col] == self.player:
                    board[row][col] = 1
                elif self.goban.ban[row][col] is not None:
                    board[row][col] = -1
        return board

    def compute_reward(self,ten: Ten,captured) -> int:
        reward = 0

        if captured:
            reward += 10 * len(captured)

        new_group_liberties = self.goban.kokyū_ten(ten)

        if len(new_group_liberties) == 1:
            reward -= 5
        elif len(new_group_liberties) >= 2:
            reward += len(new_group_liberties)

        adjacent_positions = self.goban.shūi(ten)
        for pos in adjacent_positions:
            if self.goban.ban[pos.row][pos.col] is not None and self.goban.ban[pos.row][pos.col] != self.player:
                try:
                    enemy_liberties = self.goban.kokyū_ten(pos)
                    if len(enemy_liberties) == 1:
                        reward += 3
                except KūtenError:
                    continue

        return reward

    def get_state(self) -> np.ndarray:
        # Create a state representation with three channels:
        # 0: empty positions, 1: player's stones, -1: opponent's stones
        board = np.array(self.goban.ban, dtype=object)
        player_board = (board == self).astype(np.float32)
        opponent_board = ((board != self) & (board != None)).astype(np.float32)
        empty_board = (board == None).astype(np.float32)

        state = np.stack([empty_board, player_board, opponent_board], axis=-1)
        return state.flatten()