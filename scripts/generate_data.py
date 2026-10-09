#!/usr/bin/env python3
"""Generate input samples, not fabricated experiment results."""
import argparse
from pathlib import Path
import numpy as np
from rato.utils.config import ROOT, load_config
from rato.environment.rato_env import RATOEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--output', type=Path, default=ROOT/'data/generated/scenarios.npz')
    args = p.parse_args()
    c = load_config(); c['environment']['seed'] = args.seed
    env = RATOEnv(c)
    fields = ['data_bytes','cycles','deadline_s','energy_budget_j']
    samples = {name: [] for name in fields}
    samples['channel_gains'] = []
    for i in range(args.samples):
        env.physical.sample_step()
        for name in fields: samples[name].append(env.physical.tasks[name].copy())
        samples['channel_gains'].append(env.physical.channel_gains.copy())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**{k:np.stack(v) for k,v in samples.items()},
                        association=env.physical.associations,seed=args.seed)
    print('Generated:',args.output)
if __name__ == '__main__': main()
