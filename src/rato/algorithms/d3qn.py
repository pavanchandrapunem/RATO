"""Factored multi-head Dueling Double DQN over discretized hybrid RATO actions.

The paper does not specify joint discrete-action factorization; this is an
explicit scalable approximation instead of enumerating 17^(2M)*3^M*K^M.
"""
from copy import deepcopy
import numpy as np
import torch
from .networks import DuelingQ
from .replay_buffer import ReplayBuffer

def valid_q(q, K):
    # dims [B, M, 4, 17] with valid cardinalities [17,17,3,K]
    mask = torch.arange(q.shape[-1], device=q.device)
    masks = torch.stack([mask < 17, mask < 17, mask < 3, mask < K])
    return q.masked_fill(~masks[None, None], -1e9)

def indices_to_actions(index, K, levels=17):
    ids = np.asarray(index, dtype=int)
    M = ids.shape[0]
    a = np.zeros((M, 5 + K), dtype=np.float32)
    a[:, 0] = ids[:, 0] / (levels - 1)
    a[:, 1] = np.clip(ids[:, 1] / (levels - 1), 0.001, 1)
    a[np.arange(M), 2 + ids[:, 2]] = 1.0
    a[np.arange(M), 5 + ids[:, 3]] = 1.0
    return a

class D3QN:
    def __init__(self, obs_dim, num_ues, num_edges, cfg, device, seed=0):
        self.M, self.K, self.device = num_ues, num_edges, device
        self.cfg = cfg['assumptions']['rl']; p = cfg['paper']['reinforcement_learning']
        self.levels = int(self.cfg['dqn_levels'])
        if self.levels != 17: raise ValueError('The implementation currently requires 17 levels')
        self.discount = p['discount_factor']; self.tau = p['soft_update_tau']
        self.batch_size = p['batch_size']
        self.online = DuelingQ(num_ues * obs_dim, num_ues, self.cfg['critic_hidden'],self.levels).to(device)
        self.target = deepcopy(self.online)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=self.cfg['critic_learning_rate'])
        self.buffer = ReplayBuffer(p['replay_capacity'], seed)
        self.rng = np.random.default_rng(seed)
    def act(self, obs, epsilon=0.0):
        with torch.no_grad():
            s = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
            ids = valid_q(self.online(s), self.K).argmax(-1)[0].cpu().numpy()
        for m in range(self.M):
            if self.rng.random() < epsilon:
                ids[m] = [self.rng.integers(17), self.rng.integers(17),
                          self.rng.integers(3), self.rng.integers(self.K)]
        self._last_indices = ids.copy()
        return indices_to_actions(ids, self.K, self.levels)
    def add(self, obs, action, rewards, next_obs, done):
        # Recover discretized actions from raw executed-proposal encoding.
        indexes = np.stack([np.rint(action[:,0] * 16).astype(int),
                            np.rint(action[:,1] * 16).astype(int),
                            np.argmax(action[:,2:5],axis=1),
                            np.argmax(action[:,5:],axis=1)],axis=1).astype(np.float32)
        self.buffer.add(obs, indexes, rewards, next_obs, done)
    def update(self):
        if len(self.buffer) < max(self.cfg['min_buffer_size'],self.batch_size): return None
        losses = []
        for _ in range(self.cfg['gradient_steps_per_episode']):
            s, idx, r, ns, done = self.buffer.sample(self.batch_size, self.device)
            idx = idx.long()
            batch = s.shape[0]
            qs = self.online(s).gather(-1, idx.unsqueeze(-1)).squeeze(-1).mean((1,2))
            with torch.no_grad():
                next_ids = valid_q(self.online(ns),self.K).argmax(-1)
                next_q = self.target(ns).gather(-1, next_ids.unsqueeze(-1)).squeeze(-1).mean((1,2))
                target = r.mean(1) + self.discount * (1 - done.reshape(-1)) * next_q
            loss = torch.nn.functional.mse_loss(qs, target)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg['grad_clip_norm'])
            self.optim.step()
            with torch.no_grad():
                for p,t in zip(self.online.parameters(),self.target.parameters()): t.lerp_(p,self.tau)
            losses.append(loss.item())
        return dict(q_loss=float(np.mean(losses)))
    def state_dict(self):
        return dict(online=self.online.state_dict(), target=self.target.state_dict(), optim=self.optim.state_dict())
    def load_state_dict(self, data):
        for key in ['online','target','optim']: getattr(self, key).load_state_dict(data[key])
