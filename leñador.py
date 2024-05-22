import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Optional
from atarigon.api import Goshi, Goban, Ten

class RandomForestAgent(Goshi):
    """Agent that uses a Random Forest to learn to play AtariGoN."""

    def __init__(self, name='Leñador', n_estimators=100):
        """Initializes the Random Forest agent."""
        super().__init__(name)
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        self.experiences = []
        self.encoder = LabelEncoder()

    def decide(self, goban: 'Goban') -> Optional[Ten]:
        """Decides the next move based on the Random Forest model."""
        state = self._get_state(goban)
        possible_actions = self._get_possible_actions(goban)
        
        if len(possible_actions) == 0:
            return None

        if len(self.experiences) < 100:  # Use random moves until we have enough data to train the model
            action = random.choice(possible_actions)
        else:
            state_action_pairs = [self._state_action_pair(state, action) for action in possible_actions]
            predicted_rewards = self.model.predict(state_action_pairs)
            best_action = possible_actions[np.argmax(predicted_rewards)]
            action = best_action

        # Save the state and action for learning
        self.last_state = state
        self.last_action = action

        return action

    def learn(self, state, action, reward, next_state):
        """Updates the model based on the agent's experience."""
        if state is None or action is None:
            return  # No learning if no valid state-action pair

        state_action_pair = self._state_action_pair(state, action)
        self.experiences.append((state_action_pair, reward))

        # Train the model with the collected experiences
        if len(self.experiences) > 100:  # Train only if we have enough experiences
            X, y = zip(*self.experiences)
            X = np.array(X)
            y = np.array(y)
            self.model.fit(X, y)

    def _get_state(self, goban: 'Goban'):
        """Converts the board state to a 1D array for the model."""
        return np.array(goban.ban).flatten()

    def _get_possible_actions(self, goban: 'Goban'):
        """Returns a list of all possible actions (empty positions) on the board."""
        return [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]

    def _state_action_pair(self, state, action):
        """Converts a state-action pair to a feature vector for the model."""
        return np.append(state, [action.row, action.col])