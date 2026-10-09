import numpy as np

def execution(cycles, cpu_hz, capacitance):
    cycles = np.asarray(cycles, dtype=np.float64)
    freq = np.maximum(np.asarray(cpu_hz, dtype=np.float64), 1.0)
    return cycles / freq, cycles * float(capacitance) * freq ** 2
