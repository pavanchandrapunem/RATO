import numpy as np

def transmit_time(data_bytes, rate_bps):
    bits = np.asarray(data_bytes, dtype=np.float64) * 8.0
    rate = np.asarray(rate_bps, dtype=np.float64)
    return np.where(bits > 0, bits / np.maximum(rate, 1e-20), 0.0)

def transmit_energy(power_w, time_s):
    return np.asarray(power_w) * np.asarray(time_s)
