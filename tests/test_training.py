import numpy as np
import torch
import pytest
from rato.environment.rato_env import RATOEnv
from rato.algorithms import make_agent

@pytest.mark.parametrize('algorithm',['ddpg','d3qn','madqn','maddpg','dt_maddpg'])
def test_algorithm_forward_backward(cfg,algorithm):
    torch.set_num_threads(1)
    env=RATOEnv(cfg)
    agent=make_agent(algorithm,env.observation_dim,env.M,env.K,cfg,torch.device('cpu'),3)
    obs=env.reset(1)
    for step in range(3):
        action=agent.act(obs,0.5)
        assert action.shape==(env.M,env.action_dim)
        next_obs,rewards,done,info=env.step(action)
        agent.add(obs,action,rewards,next_obs,done)
        obs=next_obs
    result=agent.update()
    assert result is not None and all(np.isfinite(v) for v in result.values())
    state=agent.state_dict()
    other=make_agent(algorithm,env.observation_dim,env.M,env.K,cfg,torch.device('cpu'),3)
    other.load_state_dict(state)
