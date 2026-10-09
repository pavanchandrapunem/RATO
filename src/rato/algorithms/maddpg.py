"""Independent per-UE actors, centralized per-UE critics; CTDE MADDPG."""
from copy import deepcopy
import numpy as np
import torch
from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer
from .exploration import continuous_action

class MADDPG:
    def __init__(self, obs_dim, num_ues, num_edges, cfg, device, seed=0):
        self.M, self.K = num_ues, num_edges
        self.device = device
        self.cfg = cfg['assumptions']['rl']; paper = cfg['paper']['reinforcement_learning']
        self.discount = paper['discount_factor']; self.tau = paper['soft_update_tau']
        self.batch_size = paper['batch_size']
        self.act_dim = 5 + num_edges
        hidden_a = self.cfg['actor_hidden']; hidden_c = self.cfg['critic_hidden']
        self.actors = [Actor(obs_dim, num_edges, hidden_a).to(device) for _ in range(num_ues)]
        self.actor_targets = [deepcopy(x) for x in self.actors]
        self.critics = [Critic(num_ues * obs_dim, num_ues * self.act_dim, hidden_c).to(device)
                        for _ in range(num_ues)]
        self.critic_targets = [deepcopy(x) for x in self.critics]
        self.actor_optim = [torch.optim.Adam(x.parameters(), lr=self.cfg['actor_learning_rate']) for x in self.actors]
        self.critic_optim = [torch.optim.Adam(x.parameters(), lr=self.cfg['critic_learning_rate']) for x in self.critics]
        self.buffer = ReplayBuffer(paper['replay_capacity'], seed)
        self.rng = np.random.default_rng(seed)

    def act(self, obs, epsilon=0.0):
        cols = [continuous_action(actor, obs[m:m+1], self.rng, epsilon,
                                  self.cfg['gaussian_noise_std'], self.device)[0]
                for m, actor in enumerate(self.actors)]
        return np.stack(cols)

    def add(self, obs, action, rewards, next_obs, done):
        self.buffer.add(obs, action, rewards, next_obs, done)

    def update(self):
        if len(self.buffer) < max(self.cfg['min_buffer_size'], self.batch_size):
            return None
        losses = []
        for _ in range(self.cfg['gradient_steps_per_episode']):
            s, a, r, ns, done = self.buffer.sample(self.batch_size, self.device)
            with torch.no_grad():
                next_actions = torch.stack([actor(ns[:, m, :]) for m, actor in enumerate(self.actor_targets)], dim=1)
            for m in range(self.M):
                with torch.no_grad():
                    target = r[:, m:m+1] + self.discount * (1 - done.reshape(-1, 1)) * self.critic_targets[m](ns, next_actions)
                q = self.critics[m](s, a)
                loss_c = (q - target).square().mean()
                self.critic_optim[m].zero_grad(set_to_none=True)
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[m].parameters(), self.cfg['grad_clip_norm'])
                self.critic_optim[m].step()
                for param in self.critics[m].parameters(): param.requires_grad_(False)
                # Other actors' actions fixed when optimizing the individual actor.
                replaced = a.detach().clone()
                replaced[:, m, :] = self.actors[m](s[:, m, :])
                loss_a = -self.critics[m](s, replaced).mean()
                self.actor_optim[m].zero_grad(set_to_none=True)
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[m].parameters(), self.cfg['grad_clip_norm'])
                self.actor_optim[m].step()
                for param in self.critics[m].parameters(): param.requires_grad_(True)
                with torch.no_grad():
                    for src, tgt in ((self.actors[m], self.actor_targets[m]),
                                     (self.critics[m], self.critic_targets[m])):
                        for p, t in zip(src.parameters(), tgt.parameters()):
                            t.lerp_(p, self.tau)
                losses.append([loss_a.item(), loss_c.item()])
        return dict(actor_loss=float(np.mean([x[0] for x in losses])),
                    critic_loss=float(np.mean([x[1] for x in losses])))

    def state_dict(self):
        return dict(actors=[a.state_dict() for a in self.actors],
                    actor_targets=[a.state_dict() for a in self.actor_targets],
                    critics=[c.state_dict() for c in self.critics],
                    critic_targets=[c.state_dict() for c in self.critic_targets],
                    actor_optim=[o.state_dict() for o in self.actor_optim],
                    critic_optim=[o.state_dict() for o in self.critic_optim])
    def load_state_dict(self, data):
        for name in self.state_dict():
            for model, state in zip(getattr(self, name), data[name]):
                model.load_state_dict(state)
