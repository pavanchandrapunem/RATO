import numpy as np

def generate_tasks(rng, count, assumptions, fixed_size_mb=None):
    task = assumptions['tasks']
    mb = (np.full(count, float(fixed_size_mb)) if fixed_size_mb is not None else
          rng.uniform(*task['size_mb_range'], size=count))
    cycles_per_mb = rng.uniform(*task['cycles_per_mb_range'], size=count)
    return dict(data_bytes=mb * 1e6, cycles=mb * cycles_per_mb,
                deadline_s=rng.uniform(*task['deadline_s_range'], size=count),
                energy_budget_j=rng.uniform(*task['energy_budget_j_range'], size=count))
