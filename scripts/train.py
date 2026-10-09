#!/usr/bin/env python3
"""Train any of five provided algorithms with timestamped experiment records."""
import argparse
import json
import time
import numpy as np
from rato.algorithms import make_agent, ALGORITHMS
from rato.environment.rato_env import RATOEnv
from rato.utils.config import ROOT, load_config
from rato.utils.seeding import seed_everything
from rato.utils.device import select_device
from rato.utils.checkpoint import save_checkpoint, load_checkpoint
from rato.utils.logging_utils import create_run_dir, save_metadata, write_csv

def run(args):
    c = load_config()
    c['training']['seed'] = args.seed
    if args.episodes is not None: c['training']['episodes'] = args.episodes
    if args.steps is not None:
        c['training']['max_steps'] = args.steps
        c['environment']['max_steps'] = args.steps
    else:
        c['environment']['max_steps'] = c['training']['max_steps']
    if args.num_ues is not None: c['paper']['network']['num_ues'] = args.num_ues
    if args.num_edge is not None: c['paper']['network']['num_edge_servers'] = args.num_edge
    if args.batch_size is not None:
        c['paper']['reinforcement_learning']['batch_size'] = args.batch_size
        c['assumptions']['rl']['min_buffer_size'] = args.batch_size
    if args.device is not None: c['training']['device'] = args.device
    if args.mode is not None: c['environment']['mode'] = args.mode
    if args.algorithm == 'maddpg':
        c['environment']['use_digital_twin'] = False
        c['environment']['load_balance_penalty'] = False
    seed_everything(args.seed)
    device = select_device(c['training']['device'])
    env = RATOEnv(c)
    agent = make_agent(args.algorithm, env.observation_dim, env.M,env.K,c,device,args.seed)
    paper = c['paper']['reinforcement_learning']
    epsilon = paper['epsilon_start']
    run_dir = create_run_dir(ROOT,args.algorithm,args.seed)
    save_metadata(run_dir,c,ROOT)
    rows = []
    first_episode = 0
    if args.resume is not None:
        state = load_checkpoint(args.resume, agent, device)
        epsilon = float(state['epsilon']); first_episode = int(state['episode'])
        extra = state.get('extra',{})
        for item in extra.get('replay_items',[]): agent.buffer.items.append(item)
        if 'exploration_rng' in extra: agent.rng.bit_generator.state = extra['exploration_rng']
        if 'buffer_rng' in extra: agent.buffer.rng.bit_generator.state = extra['buffer_rng']
    started = time.perf_counter()
    best = -float('inf')
    def checkpoint(filename, episode):
        extra = dict(replay_items=list(agent.buffer.items),
                     exploration_rng=agent.rng.bit_generator.state,
                     buffer_rng=agent.buffer.rng.bit_generator.state)
        save_checkpoint(run_dir/'checkpoints'/filename,agent,episode,epsilon,extra)
    for ep in range(first_episode, c['training']['episodes']):
        obs = env.reset(args.seed + ep)
        infos = []
        loss = None
        for step in range(c['training']['max_steps']):
            action = agent.act(obs,epsilon)
            nxt, reward, done, info = env.step(action)
            agent.add(obs,action,reward,nxt,done)
            obs = nxt
            infos.append(info)
            if done: break
        loss = agent.update()
        # Decay once per *episode* in this implementation; see docs/assumptions.md.
        epsilon = max(paper['epsilon_end'], epsilon * paper['epsilon_decay'])
        summary = dict(episode=ep+1,epsilon=epsilon,
                       reward=float(np.mean([i['reward'] for i in infos])),
                       latency_s=float(np.mean([i['latency_s'] for i in infos])),
                       energy_j=float(np.mean([i['energy_j'] for i in infos])),
                       balance=float(np.mean([i['balance'] for i in infos])),
                       success_rate=float(np.mean([i['success_rate'] for i in infos])),
                       qos=float(np.mean([i['qos'] for i in infos])))
        rows.append(summary)
        if summary['reward'] > best:
            best = summary['reward']; checkpoint('best_training_reward.pt',ep+1)
        if (ep+1) % c['training']['save_every'] == 0:
            checkpoint('latest.pt',ep+1)
        if (ep+1) % c['training']['log_interval'] == 0 or ep == first_episode:
            print(f'{args.algorithm} episode={ep+1} reward={summary["reward"]:.4f} '
                  f'success={summary["success_rate"]:.3f} epsilon={epsilon:.3f} loss={loss}',flush=True)
            write_csv(run_dir/'raw_metrics'/'training.csv',rows)
    checkpoint('latest.pt',c['training']['episodes'])
    write_csv(run_dir/'raw_metrics'/'training.csv',rows)
    result = dict(algorithm=args.algorithm,episodes=c['training']['episodes'],seed=args.seed,
                  duration_sec=round(time.perf_counter()-started,3),
                  best_training_reward=best,
                  final_training_reward=rows[-1]['reward'] if rows else None)
    (run_dir/'summary'/'metrics.json').write_text(json.dumps(result,indent=2))
    print('Saved:',run_dir)
    return run_dir

def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', choices=ALGORITHMS, default='dt_maddpg')
    p.add_argument('--episodes',type=int)
    p.add_argument('--steps',type=int)
    p.add_argument('--num-ues',type=int)
    p.add_argument('--num-edge',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=('auto','cpu','cuda'))
    p.add_argument('--mode',choices=('full','edge_only','cloud_only','no_dt_deviation'))
    p.add_argument('--resume',type=str)
    return p
if __name__ == '__main__': run(parser().parse_args())
