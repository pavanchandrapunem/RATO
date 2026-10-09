import numpy as np

def validate_actions(gamma, cluster_coeffs, destinations, num_edges):
    return (np.all((gamma >= 0) & (gamma <= 1))
            and np.all((cluster_coeffs >= 0) & (cluster_coeffs <= 1))
            and np.isclose(cluster_coeffs.sum(), 1.0)
            and np.all((destinations >= 0) & (destinations <= num_edges)))

def deadline_violation(latency, max_latency):
    return np.maximum(np.asarray(latency) - np.asarray(max_latency), 0.0)
