import pytest
import torch
import numpy as np
import tempfile
import os
from src.dqn import DQN
from src.replay_buffer import ReplayBuffer
from src.agent import DQNAgent

def test_dqn_dimensions():
    """Vérifie que le réseau produit les bonnes dimensions."""
    model = DQN(state_size=6, action_size=3)
    x = torch.randn(10, 6)  # batch de 10, 6 états
    out = model(x)
    assert out.shape == (10, 3)

def test_replay_buffer():
    """Vérifie le fonctionnement du replay buffer."""
    buffer = ReplayBuffer(capacity=10)
    
    # Ajout de transitions
    for i in range(15):
        buffer.push(
            np.zeros(6),  # state
            i % 3,        # action
            1.0,          # reward
            np.zeros(6),  # next_state
            False         # done
        )
    
    assert len(buffer) == 10  # capacité max
    
    # Échantillonnage
    batch = buffer.sample(4)
    assert len(batch) == 5  # (states, actions, rewards, next_states, dones)
    assert batch[0].shape == (4, 6)

def test_epsilon_greedy():
    """Vérifie que epsilon-greedy explore et exploite."""
    agent = DQNAgent(epsilon=1.0)
    
    # Avec epsilon=1.0, devrait explorer
    actions = [agent.act(np.zeros(6), explore=True) for _ in range(100)]
    assert len(set(actions)) > 1  # plusieurs actions différentes
    
    # Avec epsilon=0, devrait toujours prendre la même
    agent.epsilon = 0.0
    action = agent.act(np.zeros(6), explore=True)
    for _ in range(10):
        assert agent.act(np.zeros(6), explore=True) == action

def test_optimization_step():
    """Vérifie qu'une étape d'optimisation ne cause pas d'erreur."""
    agent = DQNAgent(batch_size=4)
    
    # Remplir le buffer
    for _ in range(10):
        agent.remember(
            np.random.randn(6),
            np.random.randint(0, 3),
            np.random.randn() * 10,
            np.random.randn(6),
            np.random.random() > 0.8
        )
    
    # Optimisation
    loss = agent.optimize_model()
    assert loss is not None
    assert isinstance(loss, float)

def test_target_network_sync():
    """Vérifie la synchronisation du target network."""
    agent = DQNAgent()
    
    # Modifier policy_net
    with torch.no_grad():
        for param in agent.policy_net.parameters():
            param.data += torch.randn_like(param)
    
    # Vérifier qu'ils sont différents
    policy_params = list(agent.policy_net.parameters())[0].clone()
    target_params = list(agent.target_net.parameters())[0].clone()
    assert not torch.equal(policy_params, target_params)
    
    # Synchroniser
    agent.update_target_network()
    
    # Vérifier qu'ils sont égaux
    policy_params = list(agent.policy_net.parameters())[0].clone()
    target_params = list(agent.target_net.parameters())[0].clone()
    assert torch.equal(policy_params, target_params)

def test_save_load():
    """Vérifie la sauvegarde et le chargement du modèle."""
    agent = DQNAgent()
    agent.epsilon = 0.123
    agent.step_count = 42
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        path = tmp.name
    
    try:
        agent.save(path)
        
        new_agent = DQNAgent()
        new_agent.load(path)
        
        assert new_agent.epsilon == 0.123
        assert new_agent.step_count == 42
        
        # Vérifier que les poids sont chargés
        for p1, p2 in zip(agent.policy_net.parameters(), new_agent.policy_net.parameters()):
            assert torch.equal(p1, p2)
            
    finally:
        os.unlink(path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])