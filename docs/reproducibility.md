# Reproducibility

`results/YYYY-MM-DD_HH-MM-SS_microseconds_algorithm_seed.../` holds copied config, Git commit / dirty-state metadata, checkpoints, CSV and JSON. Training seeds NumPy, Python, PyTorch; scenario RNG uses seed+episode. CPU/GPU computations need not be bitwise identical. Full resume does not guarantee bitwise equivalence, see assumptions. Lock dependencies after install: `python -m pip freeze > requirements-lock.txt`.
