import numpy as np

# Raw executed action layout: [gamma, power_bid, route probabilities (3),
#                               collaborative-edge probabilities (K)]
def action_dimension(K):
    return 5 + K

def default_actions(M, K):
    a = np.zeros((M, action_dimension(K)), dtype=np.float32)
    a[:, 0] = 0.5
    a[:, 1] = 0.5
    a[:, 2] = 1.0
    a[:, 5:] = 1.0 / K
    return a

def project_actions(raw, associations, K, mode='full', power_policy='learned'):
    """Hard decisions for the physical simulator, straight-through acts only in RL."""
    raw = np.asarray(raw, dtype=float)
    M = len(associations)
    if raw.shape != (M, action_dimension(K)):
        raise ValueError(f'Expected actions shape {(M,action_dimension(K))}, got {raw.shape}')
    raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
    gamma = np.clip(raw[:, 0], 0.0, 1.0)
    bids = np.clip(raw[:, 1], 0.001, 1.0)
    route = np.argmax(raw[:, 2:5], axis=1)
    if mode == 'edge_only':
        route[route == 2] = 0
    elif mode == 'cloud_only':
        route[:] = 2
    collaboration = np.argmax(raw[:, 5:], axis=1)
    for i in range(M):
        if collaboration[i] == associations[i]:
            collaboration[i] = (associations[i] + 1) % K
    clusters = np.zeros(K)
    for k in range(K):
        ids = np.flatnonzero(associations == k)
        clusters[k] = bids[ids].mean() if len(ids) else 0.001
    if power_policy != 'learned':
        idx = np.arange(1, K + 1, dtype=float)
        if power_policy == 'equal': clusters = np.ones(K)
        elif power_policy == 'linear': clusters = idx
        elif power_policy == 'exponential': clusters = 2 ** (idx - 1)
        else: raise ValueError(power_policy)
    coefficients = clusters / clusters.sum()
    return dict(gamma=gamma, power_coefficients=coefficients, routes=route,
                collaboration=collaboration)
