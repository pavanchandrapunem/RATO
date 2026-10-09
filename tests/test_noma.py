import numpy as np
from rato.communication.noma import noma_rates,power_allocation

def test_sic_order_and_power():
    gains=np.array([0.01,0.02,0.04])
    powers=power_allocation(1,10)
    rate=noma_rates(gains,np.array([0,0,0]),powers,20e6,2e-12)
    assert rate.shape==(3,)
    assert np.all(rate > 0) and np.all(np.isfinite(rate))
    expected = 20e6 * np.log2(1 + 0.04*10/(2e-12+(0.01+0.02)*10))
    assert np.isclose(rate[2], expected)
    for scheme in ['equal','linear','exponential']:
        assert np.isclose(power_allocation(5,10,scheme).sum(),10)
