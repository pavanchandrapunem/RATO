"""Single-agent global actor + critic DDPG baseline."""
from copy import deepcopy
import numpy as np
import torch
from .networks import CentralActor, Critic
from .replay_buffer import ReplayBuffer

class DDPG:
    def __init__(self, obs_dim, num_ues, num_edges, cfg, device, seed=0):
        self.M, self.K, self.device = num_ues, num_edges, device
        self.cfg = cfg['assumptions']['rl']; paper = cfg['paper']['reinforcement_learning']
        self.discount = paper['discount_factor']; self.tau = paper['soft_update_tau']
        self.batch_size = paper['batch_size']
        self.actor = CentralActor(obs_dim, num_ues, num_edges, self.cfg['actor_hidden']).to(device)
        self.actor_target = deepcopy(self.actor)
        self.critic = Critic(num_ues * obs_dim, num_ues * (5 + num_edges), self.cfg['critic_hidden']).to(device)
        self.critic_target = deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg['actor_learning_rate'])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg['critic_learning_rate'])
        self.buffer = ReplayBuffer(paper['replay_capacity'], seed)
        self.rng = np.random.default_rng(seed)
    def act(self, obs, epsilon=0.0):
        with torch.no_grad():
            s = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
            action = self.actor(s)[0].cpu().numpy()
        for m in range(self.M):
            if self.rng.random() < epsilon:
                action[m] += self.rng.normal(0, self.cfg['gaussian_noise_std'], size=action.shape[-1])
        return action.astype(np.float32)
    def add(self, obs, action, rewards, next_obs, done):
        self.buffer.add(obs, action, rewards, next_obs, done)
    def update(self):
        if len(self.buffer) < max(self.cfg['min_buffer_size'], self.batch_size): return None
        losses = []
        for _ in range(self.cfg['gradient_steps_per_episode']):
            s, a, r, ns, done = self.buffer.sample(self.batch_size, self.device)
            with torch.no_grad():
                targ = r.mean(dim=1, keepdim=True) + self.discount * (1 - done.reshape(-1,1)) * self.critic_target(ns, self.actor_target(ns))
            loss_c = (self.critic(s, a) - targ).square().mean()
            self.critic_optim.zero_grad(set_to_none=True)
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg['grad_clip_norm'])
            self.critic_optim.step()
            for param in self.critic.parameters(): param.requires_grad_(False)
            loss_a = -self.critic(s, self.actor(s)).mean()
            self.actor_optim.zero_grad(set_to_none=True)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg['grad_clip_norm'])
            self.actor_optim.step()
            for param in self.critic.parameters(): param.requires_grad_(True)
            with torch.no_grad():
                for src, tgt in ((self.actor,self.actor_target),(self.critic,self.critic_target)):
                    for p, t in zip(src.parameters(), tgt.parameters()): t.lerp_(p, self.tau)
            losses.append((loss_a.item(), loss_c.item()))
        return dict(actor_loss=float(np.mean([a for a,c in losses])),
                    critic_loss=float(np.mean([c for a,c in losses])))
    def state_dict(self):
        return {name: getattr(self, name).state_dict() for name in ('actor','actor_target','critic','critic_target','actor_optim','critic_optim')}
    def load_state_dict(self, data):
        for name, state in data.items(): getattr(self, name).load_state_dict(state)
