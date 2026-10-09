import numpy as np
from rato.environment.physical_system import PhysicalSystem
from rato.environment.digital_twin import DigitalTwin

def test_twin_zero_deviation(cfg):
    rng=np.random.default_rng(1)
    p=PhysicalSystem(cfg,rng)
    dt=DigitalTwin(p,rng,std=0,enabled=False)
    np.testing.assert_array_equal(dt.ue_est,dt.ue_actual)
    np.testing.assert_array_equal(dt.edge_est,dt.edge_actual)
