"""Published uplink NOMA SIC interpretation, Equations (7)–(9)."""
import numpy as np

def power_allocation(num_clusters, total_power, scheme='equal'):
    k = np.arange(1, num_clusters + 1, dtype=np.float64)
    if scheme == 'equal':
        weights = np.ones(num_clusters)
    elif scheme == 'linear':
        weights = k
    elif scheme == 'exponential':
        weights = 2.0 ** (k - 1)
    else:
        raise ValueError(scheme)
    return total_power * weights / weights.sum()

def noma_rates(gains, associations, cluster_powers, bandwidth_hz, noise_power_w):
    """Ascending |h|^2; stronger decoded first, weaker users interfere.

    The paper assigns the same p_k to each UE in a cluster; this may imply
    aggregate transmitted power > the named total p_K (see assumptions).
    """
    gains = np.asarray(gains, dtype=np.float64)
    assoc = np.asarray(associations, dtype=int)
    powers = np.asarray(cluster_powers, dtype=np.float64)
    rates = np.zeros(gains.size, dtype=np.float64)
    for k in range(powers.size):
        ids = np.flatnonzero(assoc == k)
        ids = ids[np.argsort(gains[ids], kind='stable')]
        weaker_sum = 0.0
        for m in ids:
            p = max(powers[k], 0.0)
            sinr = gains[m] * p / max(noise_power_w + weaker_sum * p, 1e-30)
            rates[m] = bandwidth_hz * np.log2(1 + sinr)
            weaker_sum += gains[m]
    return rates
