import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.dqn import DQN
from src.replay_buffer import ReplayBuffer

class DQNAgent:
    """Agent DQN avec epsilon-greedy et target network."""
    
    def __init__(
        self,
        state_size=6,
        action_size=3,
        hidden_size=64,
        buffer_capacity=50000,
        batch_size=64,
        gamma=0.99,
        learning_rate=1e-3,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        target_update_freq=10,
        device=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Réseaux
        self.policy_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Loss
        self.criterion = nn.SmoothL1Loss()
    
    def act(self, state, explore=True):
        """Choisit une action selon epsilon-greedy."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke une transition dans le buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def optimize_model(self):
        """Effectue une étape d'optimisation sur le policy network."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Échantillonner
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir en tenseurs
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q-valeurs courantes
        current_q = self.policy_net(states).gather(1, actions)
        
        # Q-valeurs cibles (Bellman)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = self.criterion(current_q, target_q)
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour la stabilité
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.step_count += 1
        
        # Mise à jour de epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Synchronisation target network
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Copie les poids du policy network vers le target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """Sauvegarde le modèle."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)
    
    def load(self, path):
        """Charge un modèle sauvegardé."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']