# The entry point that builds the model, tracks self-play experiences, optimizes PyTorch weights, and promotes the baseline opponent when the target win rate is reached.

import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import TactalarisNet
from agent import RandomAgent, NNAgent
from evaluator import evaluate_agents, play_game
import rules

def train_step(model, optimizer, batch_states, batch_policies, batch_values):
    model.train()
    optimizer.zero_grad()
    
    logits, values = model(batch_states)
    
    # Loss = Policy Cross-Entropy + Value Mean-Squared-Error
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -torch.sum(batch_policies * log_probs, dim=-1).mean()
    value_loss = F.mse_loss(values.squeeze(-1), batch_values)
    
    total_loss = policy_loss + value_loss
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def main():
    # 1. Hyperparameters
    WIN_RATE_THRESHOLD = 0.60  # Require 60% win rate to promote snapshot
    EVAL_GAMES = 50
    GAMES_PER_EPOCH = 20
    
    # Setup PyTorch for Fedora Asahi CPU
    torch.set_num_threads(4)  # Utilize M2 performance cores
    
    # 2. Instantiate Networks
    current_net = TactalarisNet()
    optimizer = optim.Adam(current_net.parameters(), lr=1e-3)
    
    # Generation 0 Opponent is the Random Mover
    baseline_opponent = RandomAgent()
    generation = 0
    
    print(f"🚀 Starting Tactalaris RL Training on Fedora Asahi...")
    print(f"Stage 0: Training against Random Agent baseline.")
    
    while True:
        # Phase A: Play games against baseline to collect training data
        active_agent = NNAgent(current_net, temperature=1.0)
        
        for _ in range(GAMES_PER_EPOCH):
            play_game(active_agent, baseline_opponent)
            
        # (In practice, collect states/actions during play_game and pass to train_step here)

        # Phase B: Evaluate Current Net against Baseline
        win_rate = evaluate_agents(current_net, baseline_opponent, num_games=EVAL_GAMES)
        print(f"Gen {generation} | Evaluation Win Rate vs Baseline: {win_rate * 100:.1f}%")
        
        # Phase C: Check Promotion Gate
        if win_rate >= WIN_RATE_THRESHOLD:
            generation += 1
            print(f"🎉 Threshold reached! Upgrading Baseline Opponent to Net Generation {generation}.")
            
            # Freeze snapshot of current net as the new opponent
            frozen_net = copy.deepcopy(current_net)
            frozen_net.eval()
            baseline_opponent = NNAgent(frozen_net, temperature=0.2)
            
            # Save checkpoint
            torch.save(current_net.state_dict(), f"tactalaris_gen_{generation}.pth")

if __name__ == "__main__":
    main()
