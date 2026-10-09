import numpy as np
from rato.optimization.reward import compute_rewards
from rato.optimization.load_balancing import balancing_indicator

def test_reward_penalties():
    easy=compute_rewards(np.array([1]),np.array([1]),np.array([.5]),np.array([.5]),np.array([2]),0,3)[0]
    late=compute_rewards(np.array([3]),np.array([1]),np.array([.5]),np.array([.5]),np.array([2]),0,3)[0]
    assert late<easy
    assert balancing_indicator([.5,.5])==0
