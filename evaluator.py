# Plays match series between two agents and calculates win rates to determine when to upgrade the fixed opponent.

import rules
from agent import RandomAgent, NNAgent

def play_game(agent_white, agent_black):
    """
    Simulates a single game between two agents until terminal state.
    """
    fen = "qcnbrrbzck/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/KCZBRRBNCQ w - - 0 1"
    fens = []
    current_player = 1  # 1 for White, -1 for Black
    result = 9999 # 1 if agent_white wins, -1 if agent_black wins, 0 for draw, 9999 for ongoing game
    
    while result == 9999:
    
        agent = agent_white if current_player == 1 else agent_black
        move = agent.get_move(fen)
        
        if move is None:
            break
        
        rules.load("fen": fen)
        rules.move("from:" move.from, "to": move.to, "promotion": move.promotion)
        
        fen = rules.fen()
        fens.append(fen)

        if rules.fiftyMoves() or rules.insufficientMaterial() or rules.inStalemate() or has_three_occurrences(fens):
        
            result = 0
            
        if rules.inCheck():
        
            if rules.inCheckmate():
            
                if rules.turn() == "b":
                
                    result = 1
                    
                else
                
                    result = -1        

        current_player *= -1

    return result

def evaluate_agents(candidate_net, baseline_agent, num_games=100):
    """
    Runs a tournament where candidate_net plays half games as White and half as Black.
    """
    candidate_agent = NNAgent(candidate_net, temperature=0.0) # Greedy for evaluation
    candidate_wins = 0
    
    for i in range(num_games):
        if i % 2 == 0:
            winner = play_game(candidate_agent, baseline_agent)
            if winner == 1:
                candidate_wins += 1
        else:
            winner = play_game(baseline_agent, candidate_agent)
            if winner == -1:
                candidate_wins += 1
                
    win_rate = candidate_wins / num_games
    return win_rate
    
def has_three_occurrences(arr):

    # helper function to detect threefold repetition
    # check if any element occurs at least three times in a list (array)
    
    counts = {}
    for item in arr:
        # Increment count for the item (default to 0 if it doesn't exist yet)
        counts[item] = counts.get(item, 0) + 1
        
        # Check if we hit 3 occurrences
        if counts[item] == 3:
            return True
            
    return False
