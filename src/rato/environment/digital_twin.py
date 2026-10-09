import numpy as np

class DigitalTwin:
    """Cloud-hosted virtual replica of simulated CPU capacities."""
    def __init__(self, physical, rng, std=0.02, enabled=True):
        self.physical = physical
        self.rng = rng
        self.std = std if enabled else 0.0
        self.sync()

    def sync(self):
        p = self.physical
        self.ue_est = p.ue_freq_hz.copy()
        self.edge_est = p.edge_freq_hz.copy()
        self.cloud_est = float(p.cloud_freq_hz)
        self.ue_delta = self.ue_est * self.rng.normal(0, self.std, size=p.M)
        self.edge_delta = self.edge_est * self.rng.normal(0, self.std, size=p.K)
        self.cloud_delta = self.cloud_est * float(self.rng.normal(0, self.std))
        self.ue_actual = np.maximum(self.ue_est + self.ue_delta, 1e6)
        self.edge_actual = np.maximum(self.edge_est + self.edge_delta, 1e6)
        self.cloud_actual = max(self.cloud_est + self.cloud_delta, 1e6)
