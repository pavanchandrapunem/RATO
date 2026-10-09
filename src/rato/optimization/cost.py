import numpy as np

def dynamic_weights(deadline, energy_budget):
    """Assumption: normalized inverse tolerances, both dimensionless."""
    a = 1 / np.maximum(np.asarray(deadline, dtype=float) / 8.0, 1e-8)
    b = 1 / np.maximum(np.asarray(energy_budget, dtype=float) / 25.0, 1e-8)
    return a / (a + b), b / (a + b)

def weighted_cost(latency, energy, wt, we):
    return np.asarray(wt) * np.asarray(latency) + np.asarray(we) * np.asarray(energy)
