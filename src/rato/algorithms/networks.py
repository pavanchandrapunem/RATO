"""Small fully-connected actor/critic networks; sizes are gap-filling assumptions."""
import torch
from torch import nn
import torch.nn.functional as F

def mlp(dim_in, hidden, dim_out):
    layers = []
    for size in hidden:
        layers += [nn.Linear(dim_in, size), nn.ReLU()]
        dim_in = size
    layers.append(nn.Linear(dim_in, dim_out))
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, edge_count, hidden=(64, 64)):
        super().__init__()
        self.K = edge_count
        self.net = mlp(obs_dim, hidden, 5 + edge_count)
    def forward(self, x):
        z = self.net(x)
        return torch.cat([torch.sigmoid(z[..., :2]),
                          F.softmax(z[..., 2:5], dim=-1),
                          F.softmax(z[..., 5:], dim=-1)], dim=-1)

class CentralActor(nn.Module):
    def __init__(self, obs_dim, num_ues, num_edges, hidden=(64, 64)):
        super().__init__()
        self.M = num_ues
        self.K = num_edges
        self.net = mlp(obs_dim * num_ues, hidden, num_ues * (5 + num_edges))
    def forward(self, x):
        out = self.net(x.reshape(x.size(0), -1))
        out = out.reshape(-1, self.M, 5 + self.K)
        return torch.cat([torch.sigmoid(out[..., :2]),
                          F.softmax(out[..., 2:5], dim=-1),
                          F.softmax(out[..., 5:], dim=-1)], dim=-1)

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, hidden, 1)
    def forward(self, obs, action):
        return self.net(torch.cat([obs.reshape(obs.size(0), -1),
                                   action.reshape(action.size(0), -1)], dim=-1))

class DuelingQ(nn.Module):
    """Multi-head factored Double-Dueling Q approximation, 17 levels/head."""
    def __init__(self, input_dim, num_ues, hidden=(128, 128), levels=17):
        super().__init__()
        self.M, self.levels = num_ues, levels
        self.encoder = mlp(input_dim, hidden, hidden[-1])
        self.value = nn.Linear(hidden[-1], num_ues * 4)
        self.advantage = nn.Linear(hidden[-1], num_ues * 4 * levels)
    def forward(self, x):
        h = self.encoder(x.reshape(x.size(0), -1))
        v = self.value(h).reshape(-1, self.M, 4, 1)
        adv = self.advantage(h).reshape(-1, self.M, 4, self.levels)
        return v + adv - adv.mean(dim=-1, keepdim=True)
