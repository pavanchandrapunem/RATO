"""An explicitly assumed equal-bandwidth OMA comparison, not author code."""
import numpy as np

def oma_rates(gains, associations, cluster_powers, bandwidth_hz, noise_power_w):
    gains = np.asarray(gains, dtype=np.float64)
    assoc = np.asarray(associations, dtype=int)
    powers = np.asarray(cluster_powers, dtype=np.float64)
    rates = np.zeros(gains.size, dtype=np.float64)
    for k in range(len(powers)):
        ids = np.flatnonzero(assoc == k)
        if len(ids):
            bandwidth = bandwidth_hz / len(ids)
            rates[ids] = bandwidth * np.log2(1 + gains[ids] * powers[k] / max(noise_power_w, 1e-30))
    return rates
