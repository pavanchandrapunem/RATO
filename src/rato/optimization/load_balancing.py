import numpy as np

def utilization(assigned_cycles, capacity_cycles):
    return np.asarray(assigned_cycles, dtype=np.float64) / np.maximum(capacity_cycles, 1e-20)

def balancing_indicator(ratios):
    return float(np.var(np.asarray(ratios, dtype=np.float64)))
