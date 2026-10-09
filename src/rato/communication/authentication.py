"""Simulation of authorization and its resource costs; NOT real cryptography."""
class AuthenticationModel:
    def __init__(self, latency_s=0.005, energy_j=0.002):
        self.latency_s = float(latency_s)
        self.energy_j = float(energy_j)
        self.trusted = True

    def authenticate(self, ue, dest_edge):
        if ue < 0 or dest_edge < 0:
            return False, 0.0, 0.0
        return self.trusted, self.latency_s, self.energy_j
