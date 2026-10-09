import numpy as np

def mean_metrics(infos):
    fields = ('reward', 'latency_s', 'energy_j', 'balance', 'success_rate', 'qos', 'deadline_violation_rate', 'throughput_mbps')
    return {f: float(np.mean([x[f] for x in infos])) for f in fields}
