import numpy as np
from ..communication.channel import sample_channel_gains
from .task_generator import generate_tasks

class PhysicalSystem:
    """Synthetic topology and per-slot workload; no physical hardware devices required."""
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        p = cfg['paper']; a = cfg['assumptions']; e = cfg['environment']
        self.M = p['network']['num_ues']; self.K = p['network']['num_edge_servers']
        self.associations = rng.integers(0, self.K, size=self.M)
        # Guarantee an edge cluster exists for every ES where M>=K.
        if self.M >= self.K:
            self.associations[:self.K] = np.arange(self.K)
            rng.shuffle(self.associations)
        r = float(a['network']['ue_location_radius_m']) * np.sqrt(rng.random(self.M))
        t = rng.uniform(0, 2 * np.pi, self.M)
        self.locations_xy_m = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
        self.ue_freq_hz = rng.uniform(*p['computation']['ue_freq_ghz'], size=self.M) * 1e9
        self.edge_freq_hz = rng.uniform(*p['computation']['edge_freq_ghz'], size=self.K) * 1e9
        self.cloud_freq_hz = p['computation']['cloud_freq_ghz'] * 1e9
        self.sample_step()

    def sample_step(self):
        p = self.cfg['paper']['communication']
        e = self.cfg['environment']
        self.tasks = generate_tasks(self.rng, self.M, self.cfg['assumptions'], e['force_task_size_mb'])
        self.channel_gains = sample_channel_gains(self.rng, self.M,
                                                  p['channel_gain_db_min'],p['channel_gain_db_max'])
