import numpy as np
from rato.environment.rato_env import RATOEnv
from rato.environment.action_handler import default_actions

def test_environment_shapes_and_metrics(cfg):
    env=RATOEnv(cfg)
    obs=env.reset(12)
    assert obs.shape==(3,env.observation_dim)
    act=default_actions(3,2)
    nxt,reward,done,info=env.step(act)
    assert reward.shape==(3,) and nxt.shape==obs.shape and not done
    for key in ['reward','latency_s','energy_j','balance','success_rate','qos']:
        assert np.isfinite(info[key]),(key,info[key])
    assert info['energy_j']>=0
    assert env.step(act)[2]
