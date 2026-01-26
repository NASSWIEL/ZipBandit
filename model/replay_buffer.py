"""
Replay Buffer for Experience Replay in Contextual Bandit RL.

This allows the agent to:
1. Store past experiences (state, action, reward)
2. Sample mini-batches for training
3. Prevent catastrophic forgetting
4. Improve sample efficiency
"""

import numpy as np
import torch
import os
import pickle
from collections import deque


class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling experiences.
    """
    
    def __init__(self, capacity=5000, state_dim=1024, action_dim=256):
        """
        Initialize replay buffer.
        
        EXPERT FIX: Updated action_dim from 100 to 256 for better representation.
        
        Args:
            capacity (int): Maximum number of experiences to store.
            state_dim (int): Dimension of state vectors (1024 for SONAR).
            action_dim (int): Dimension of action vectors (256 for prompts).
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Storage
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.metadata = deque(maxlen=capacity)  # Store prompt_idx and other info
        
        self.size = 0
    
    def add(self, state, action, reward, metadata=None):
        """
        Add an experience to the buffer.
        
        Args:
            state (np.ndarray): State vector (state_dim,)
            action (np.ndarray): Action vector (action_dim,)
            reward (float): Reward scalar
            metadata (dict): Optional metadata (e.g., {'prompt_idx': 56550})
        """
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.metadata.append(metadata if metadata is not None else {})
        
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, diversity_sampling=False):
        """
        Sample a mini-batch of experiences with optional diversity-aware sampling.
        
        Args:
            batch_size (int): Number of experiences to sample.
            diversity_sampling (bool): If True, prioritize diverse actions.
            
        Returns:
            Tuple of (states, actions, rewards, metadata_list) as torch tensors + list.
        """
        if self.size < batch_size:
            batch_size = self.size
        
        if diversity_sampling and self.size > batch_size * 2:
            # CRITICAL FIX: Diversity-aware sampling
            # Sample more candidates and select most diverse subset
            candidate_size = min(batch_size * 4, self.size)
            candidate_indices = np.random.choice(self.size, candidate_size, replace=False)
            
            # Get candidate actions
            candidate_actions = np.array([self.actions[i] for i in candidate_indices])
            
            # Select diverse subset using greedy maximum distance selection
            selected_indices = [candidate_indices[0]]  # Start with random
            remaining = set(candidate_indices) - {candidate_indices[0]}
            
            while len(selected_indices) < batch_size and remaining:
                # Find action with maximum minimum distance to selected
                selected_actions = np.array([self.actions[i] for i in selected_indices])
                max_min_dist = -1
                best_idx = None
                
                for idx in remaining:
                    action = self.actions[idx]
                    # Compute minimum distance to any selected action
                    dists = np.linalg.norm(selected_actions - action, axis=1)
                    min_dist = np.min(dists)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining.remove(best_idx)
                else:
                    break
            
            indices = np.array(selected_indices)
        else:
            # Standard uniform random sampling
            indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Convert to numpy arrays first to avoid slow tensor creation
        states_np = np.array([self.states[i] for i in indices])
        actions_np = np.array([self.actions[i] for i in indices])
        rewards_np = np.array([self.rewards[i] for i in indices])
        metadata_list = [self.metadata[i] for i in indices]
        
        states = torch.FloatTensor(states_np)
        actions = torch.FloatTensor(actions_np)
        rewards = torch.FloatTensor(rewards_np)
        
        return states, actions, rewards, metadata_list
    
    def __len__(self):
        return self.size
    
    def save(self, path):
        """Save buffer to disk."""
        data = {
            'states': list(self.states),
            'actions': list(self.actions),
            'rewards': list(self.rewards),
            'metadata': list(self.metadata),
            'size': self.size,
            'capacity': self.capacity,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Replay buffer saved to {path}")
    
    def load(self, path):
        """Load buffer from disk."""
        if not os.path.exists(path):
            print(f"Warning: Replay buffer file not found: {path}")
            return False
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.capacity = data['capacity']
        self.state_dim = data['state_dim']
        self.action_dim = data['action_dim']
        self.size = data['size']
        
        self.states = deque(data['states'], maxlen=self.capacity)
        self.actions = deque(data['actions'], maxlen=self.capacity)
        self.rewards = deque(data['rewards'], maxlen=self.capacity)
        
        # Handle backward compatibility
        if 'metadata' in data:
            self.metadata = deque(data['metadata'], maxlen=self.capacity)
        else:
            # Old buffer without metadata - fill with empty dicts
            self.metadata = deque([{} for _ in range(self.size)], maxlen=self.capacity)
        
        print(f"Replay buffer loaded from {path} (size={self.size})")
        return True


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer that samples high-reward experiences more frequently.
    """
    
    def __init__(self, capacity=1000, state_dim=1024, action_dim=100, alpha=0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity (int): Maximum number of experiences to store.
            state_dim (int): Dimension of state vectors.
            action_dim (int): Dimension of action vectors.
            alpha (float): Priority exponent (0 = uniform, 1 = fully prioritized).
        """
        super().__init__(capacity, state_dim, action_dim)
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(self, state, action, reward):
        """Add experience with priority based on reward."""
        super().add(state, action, reward)
        
        # Priority = (reward + epsilon)^alpha to ensure non-zero probability
        priority = (reward + 0.01) ** self.alpha
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size):
        """Sample experiences with probability proportional to priority."""
        if self.size < batch_size:
            batch_size = self.size
        
        # Compute sampling probabilities
        priorities = np.array(list(self.priorities))
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        
        states = torch.FloatTensor([self.states[i] for i in indices])
        actions = torch.FloatTensor([self.actions[i] for i in indices])
        rewards = torch.FloatTensor([self.rewards[i] for i in indices])
        
        return states, actions, rewards
    
    def save(self, path):
        """Save prioritized buffer to disk."""
        data = {
            'states': list(self.states),
            'actions': list(self.actions),
            'rewards': list(self.rewards),
            'priorities': list(self.priorities),
            'size': self.size,
            'capacity': self.capacity,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'alpha': self.alpha,
            'max_priority': self.max_priority
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Prioritized replay buffer saved to {path}")
    
    def load(self, path):
        """Load prioritized buffer from disk."""
        if not os.path.exists(path):
            print(f"Warning: Replay buffer file not found: {path}")
            return False
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.capacity = data['capacity']
        self.state_dim = data['state_dim']
        self.action_dim = data['action_dim']
        self.size = data['size']
        self.alpha = data['alpha']
        self.max_priority = data['max_priority']
        
        self.states = deque(data['states'], maxlen=self.capacity)
        self.actions = deque(data['actions'], maxlen=self.capacity)
        self.rewards = deque(data['rewards'], maxlen=self.capacity)
        self.priorities = deque(data['priorities'], maxlen=self.capacity)
        
        print(f"Prioritized replay buffer loaded from {path} (size={self.size})")
        return True
