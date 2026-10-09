from .ddpg import DDPG
from .d3qn import D3QN
from .madqn import MADQN
from .maddpg import MADDPG
from .dt_maddpg import DTMADDPG

ALGORITHMS = {'ddpg':DDPG, 'd3qn':D3QN, 'madqn':MADQN,
              'maddpg':MADDPG, 'dt_maddpg':DTMADDPG}

def make_agent(name, obs_dim, M, K, config, device, seed=0):
    if name not in ALGORITHMS: raise ValueError(f'Unknown algorithm {name}')
    return ALGORITHMS[name](obs_dim,M,K,config,device,seed)
