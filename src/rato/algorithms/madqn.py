"""Per-UE factored multi-head DQN, with a separate network per user."""
from copy import deepcopy
import numpy as np
import torch
from .networks import DuelingQ
from .d3qn import valid_q, indices_to_actions
from .replay_buffer import ReplayBuffer

class MADQN:
    def __init__(self, obs_dim, num_ues, num_edges, cfg, device, seed=0):
        self.M, self.K, self.device = num_ues, num_edges, device
        self.cfg = cfg['assumptions']['rl']; p = cfg['paper']['reinforcement_learning']
        self.batch_size = p['batch_size']; self.discount = p['discount_factor']; self.tau = p['soft_update_tau']
        self.online = [DuelingQ(obs_dim, 1, self.cfg['critic_hidden'],17).to(device) for _ in range(num_ues)]
        self.target = [deepcopy(x) for x in self.online]
        self.optims = [torch.optim.Adam(x.parameters(), lr=self.cfg['critic_learning_rate']) for x in self.online]
        self.buffer = ReplayBuffer(p['replay_capacity'], seed)
        self.rng = np.random.default_rng(seed)
    def act(self, obs, epsilon=0.0):
        indices = []
        for m, net in enumerate(self.online):
            with torch.no_grad():
                state = torch.as_tensor(obs[m:m+1],dtype=torch.float32,device=self.device)
                idx = valid_q(net(state),self.K).argmax(-1)[0,0].cpu().numpy()
            if self.rng.random() < epsilon:
                idx = np.array([self.rng.integers(17),self.rng.integers(17),
                                self.rng.integers(3),self.rng.integers(self.K)])
            indices.append(idx)
        return indices_to_actions(np.stack(indices), self.K)
    def add(self, obs, action, rewards, next_obs, done):
        indexes = np.stack([np.rint(action[:,0] * 16).astype(int),
                            np.rint(action[:,1] * 16).astype(int),
                            np.argmax(action[:,2:5],axis=1),
                            np.argmax(action[:,5:],axis=1)],axis=1).astype(np.float32)
        self.buffer.add(obs, indexes, rewards, next_obs, done)
    def update(self):
        if len(self.buffer) < max(self.cfg['min_buffer_size'], self.batch_size): return None
        losses = []
        for _ in range(self.cfg['gradient_steps_per_episode']):
            s, idx, r, ns, done = self.buffer.sample(self.batch_size, self.device)
            idx = idx.long()
            for m in range(self.M):
                obs = s[:,m,:]; nextobs = ns[:,m,:]
                q = self.online[m](obs).gather(-1, idx[:,m:m+1,:].unsqueeze(-1)).squeeze(-1).mean((1,2))
                with torch.no_grad():
                    selected = valid_q(self.online[m](nextobs),self.K).argmax(-1)
                    target_q = self.target[m](nextobs).gather(-1,selected.unsqueeze(-1)).squeeze(-1).mean((1,2))
                    y = r[:,m] + self.discount * (1 - done.reshape(-1)) * target_q
                loss = torch.nn.functional.mse_loss(q, y)
                self.optims[m].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.online[m].parameters(),self.cfg['grad_clip_norm'])
                self.optims[m].step()
                with torch.no_grad():
                    for p,t in zip(self.online[m].parameters(),self.target[m].parameters()): t.lerp_(p,self.tau)
                losses.append(loss.item())
        return dict(q_loss=float(np.mean(losses)))
    def state_dict(self):
        return dict(online=[m.state_dict() for m in self.online],
                    target=[m.state_dict() for m in self.target],optims=[o.state_dict() for o in self.optims])
    def load_state_dict(self, data):
        for key in ['online','target','optims']:
            for obj, state in zip(getattr(self,key),data[key]): obj.load_state_dict(state)
