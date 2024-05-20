import numpy as np
import random
from typing import Optional, List, Dict
from atarigon.api import Goshi, Goban, Ten
from keras.models import Sequential
from keras.layers import Dense

class KerasAgent(Goshi):
    def __init__(self, input_size: int, output_size: int, epsilon: float = 0.01, learning_rate: float = 0.01):
        super().__init__('KerasAgent')
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon  # Exploration rate
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.input_size,)),
            Dense(64, activation='relu'),
            Dense(self.output_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def decide(self, goban: 'Goban') -> Optional[Ten]:
        empty_positions = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(empty_positions)
        else:
            state = self.encode_state(goban)
            q_values = self.model.predict(np.array([state]))[0]
            best_index = np.argmax(q_values)
            return empty_positions[best_index]

    def learn(self, state: 'Goban', action: Ten, reward: float, next_state: Optional['Goban']) -> None:
        target = reward
        if next_state is not None:
            next_state_encoded = self.encode_state(next_state)
            target += self.learning_rate * np.max(self.model.predict(np.array([next_state_encoded])))
        state_encoded = self.encode_state(state)
        self.model.fit(np.array([state_encoded]), np.array([[target]]), epochs=1, verbose=0)

    def encode_state(self, state: 'Goban') -> List[float]:
        encoded = []
        for row in state.ban:
            for cell in row:
                if cell is None:
                    encoded.append(0)
                elif cell.color == 'black':
                    encoded.append(1)
                else:
                    encoded.append(-1)
        return encoded
