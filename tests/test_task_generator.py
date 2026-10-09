import numpy as np
from rato.environment.task_generator import generate_tasks

def test_task_splitting_conservation(cfg):
    t=generate_tasks(np.random.default_rng(3),5,cfg['assumptions'])
    gamma=np.linspace(0,1,5)
    for field in ['data_bytes','cycles']:
        np.testing.assert_allclose(t[field]*gamma+t[field]*(1-gamma),t[field])
    assert np.all(t['cycles']>0)
