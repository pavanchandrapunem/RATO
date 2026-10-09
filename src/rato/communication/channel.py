import numpy as np

def db_to_power_gain(gain_db):
    """Paper's -20 to -14 dB interpreted as channel POWER gains |h|^2."""
    return 10.0 ** (np.asarray(gain_db, dtype=np.float64) / 10.0)

def sample_channel_gains(rng, size, db_min=-20.0, db_max=-14.0):
    return db_to_power_gain(rng.uniform(db_min, db_max, size=size))
