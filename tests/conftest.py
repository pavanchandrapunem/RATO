import pytest
from rato.utils.config import load_config

@pytest.fixture
def cfg():
    c=load_config()
    c['paper']['network']['num_ues']=3
    c['paper']['network']['num_edge_servers']=2
    c['paper']['reinforcement_learning']['batch_size']=2
    c['assumptions']['rl']['min_buffer_size']=2
    c['assumptions']['rl']['actor_hidden']=[16,16]
    c['assumptions']['rl']['critic_hidden']=[16,16]
    c['environment']['max_steps']=2
    c['assumptions']['rl']['gradient_steps_per_episode']=1
    return c
