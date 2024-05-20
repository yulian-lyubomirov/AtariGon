from atarigon.api import Goshi, Goban, Ten
from typing import Optional
import random

class MCTSPlayer(Goshi):
    def __init__(self, name):
        super().__init__(name)
        self.num_simulations = 1000  # Number of simulations to run for each move

    def decide(self, goban: Goban) -> Optional[Ten]:
        root_node = Node(goban.clone(), None, self)  # Create root node with current state
        for _ in range(self.num_simulations):
            node = root_node
            # Selection phase: traverse the tree until a leaf node is reached
            while node.is_expanded():
                action = node.select_action()
                node = node.get_child(action)

            # Expansion phase: expand the selected leaf node
            if not node.is_terminal():
                action = random.choice(node.get_untried_actions())
                next_state = node.state.clone()
                next_state.place_stone(action, self)
                node = node.add_child(action, next_state)

            # Simulation phase: simulate the game from the expanded node
            while not node.is_terminal():
                action = random.choice(node.get_legal_actions())
                next_state = node.state.clone()
                next_state.place_stone(action, self)
                node = node.add_child(action, next_state)

            # Backpropagation phase: propagate the result of the simulation back to the root node
            result = node.state.evaluate(self)  # Evaluate the game result from the perspective of the player
            while node is not None:
                node.update(result)
                node = node.parent

        # Select the action with the highest visit count from the root node
        return root_node.get_most_visited_action()
    
    
    def is_terminal(self, state: 'Goban') -> bool:
        return not any(Ten(row, col) for row in range(len(state.ban)) for col in range(len(state.ban[row])) if state.ban[row][col] is None)
