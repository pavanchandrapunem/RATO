import numpy as np
from rato.communication.oma import oma_rates

def test_oma_finite():
    out=oma_rates([.01,.02],[0,0],[10],20e6,2e-12)
    assert len(out)==2 and np.all(out>0) and np.all(np.isfinite(out))
