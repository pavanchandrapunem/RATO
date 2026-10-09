import numpy as np
from rato.computation.local import execution
from rato.communication.transmission import transmit_time

def test_computation_units():
    t,e=execution(1e9,1e9,1e-27)
    assert np.isclose(t,1) and np.isclose(e,1)
    assert np.isclose(transmit_time(1000000,8e6),1)
