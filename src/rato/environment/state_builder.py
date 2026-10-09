import numpy as np
from ..optimization.cost import dynamic_weights

def build_observations(physical, dt, last_utilization, cluster_power):
    """Per-UE task + global DT/edge state as paper's shared observation concept."""
    M, K = physical.M, physical.K
    t = physical.tasks
    wt, we = dynamic_weights(t['deadline_s'], t['energy_budget_j'])
    global_features = np.concatenate([
        dt.edge_est / 12e9,
        dt.edge_delta / 12e9,
        last_utilization,
        cluster_power,
        np.array([dt.cloud_est / 100e9, dt.cloud_delta / 100e9]),
        np.bincount(physical.associations, minlength=K) / max(M, 1),
    ])
    out = []
    for m in range(M):
        local = np.array([
            t['data_bytes'][m] / 60e6, t['cycles'][m] / 9e9,
            t['deadline_s'][m] / 8, t['energy_budget_j'][m] / 25,
            wt[m], we[m], dt.ue_est[m] / 2.5e9, dt.ue_delta[m] / 2.5e9,
            physical.channel_gains[m] / 0.04, physical.associations[m] / max(K-1, 1),
        ])
        out.append(np.concatenate([local, global_features]))
    return np.stack(out).astype(np.float32)
