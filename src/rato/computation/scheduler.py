import numpy as np

def parallel_latency(local_s, remote_s):
    return np.maximum(local_s, remote_s)
