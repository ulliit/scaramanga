# Policy & Value Network
# This network accepts the tensor representation of the board and outputs both move probabilities (Policy) and state evaluations (Value).

import torch
import torch.nn as nn
import torch.nn.functional as F

class TactalarisNet(nn.Module):
    def __init__(self, input_channels=20, board_dim=10, action_space_size=10000):
        super(TactalarisNet, self).__init__()
        
        # Shared Convolutional Backbone
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Policy Head (Outputs raw logits for all possible actions)
        # Policy Head Output Size must match the move index range (10,000)
        
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * board_dim * board_dim, action_space_size)
        
        # Value Head (Outputs state evaluation between -1 and 1)
        
        self.value_conv = nn.Conv2d(128, 3, kernel_size=1)
        self.value_fc1 = nn.Linear(3 * board_dim * board_dim, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return logits, value
