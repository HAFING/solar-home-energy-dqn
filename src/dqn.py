import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Réseau neuronal pour estimer les Q-valeurs.
    
    Architecture: 6 entrées (état) -> 64 -> 64 -> 3 sorties (actions)
    """
    def __init__(self, state_size=6, action_size=3, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
