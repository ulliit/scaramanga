# Defines player instances: a random mover (baseline 0) and a Neural Network agent.

import random
import rules
from mcts import get_masked_policy, select_action

class RandomAgent:
    def get_move(self, fen):
    
        rules.load({"fen": fen})
        legal_moves = rules.get_legal_moves()
        
        if not legal_moves:
            return None
        return random.choice(legal_moves)

class NNAgent:
    def __init__(self, model, temperature=0.5):
        self.model = model
        self.temperature = temperature

    def get_move(self, fen):
        probs, _, legal_moves = get_masked_policy(self.model, fen)
        if not legal_moves:
            return None
        return select_action(probs, legal_moves, temperature=self.temperature)
