from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """One joint transition per timestep; preserves all agents' alignment."""
    def __init__(self, capacity, seed=0):
        self.items = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)
    def __len__(self): return len(self.items)
    def add(self, state, action, rewards, next_state, done):
        self.items.append(tuple(np.asarray(x, dtype=np.float32).copy()
                                for x in (state, action, rewards, next_state)) + (float(done),))
    def sample(self, n, device):
        indexes = self.rng.choice(len(self), size=n, replace=False)
        sampled = [self.items[i] for i in indexes]
        return tuple(torch.as_tensor(np.stack([row[j] for row in sampled]),
                                     dtype=torch.float32, device=device) for j in range(5))
