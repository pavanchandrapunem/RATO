import numpy as np
from .constraints import deadline_violation
from .cost import weighted_cost

def compute_rewards(latency, energy, wt, we, deadline, balance_penalty,
                    coeff_deadline, reward_scale=1.0, include_balance=True):
    cost = weighted_cost(latency, energy, wt, we)
    late = deadline_violation(latency, deadline) / np.maximum(deadline, 1e-10)
    penalty = coeff_deadline * late
    if include_balance:
        penalty = penalty + balance_penalty
    return -reward_scale * (cost + penalty)
