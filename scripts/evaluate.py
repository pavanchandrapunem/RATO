#!/usr/bin/env python3
"""Evaluate trained policy on deterministic held-out synthetic scenarios."""
import argparse
import json
from rato.algorithms import make_agent,ALGORITHMS
from rato.environment.rato_env import RATOEnv
from rato.utils.config import ROOT,load_config
from rato.utils.device import select_device
from rato.utils.checkpoint import load_checkpoint
from rato.utils.logging_utils import create_run_dir,save_metadata,write_csv
from rato.utils.metrics import mean_metrics

def evaluate(agent, env, seed=10000, episodes=10):
    events=[]
    for ep in range(episodes):
        obs = env.reset(seed+ep)
        done=False
        while not done:
            action = agent.act(obs,0.0)
            obs,reward,done,info = env.step(action)
            events.append(dict(episode=ep+1,step=env.step_count,**{
                k:v for k,v in info.items() if isinstance(v,(float,int))}))
    return events,mean_metrics(events)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algorithm',choices=ALGORITHMS,default='dt_maddpg')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--mode',choices=('full','edge_only','cloud_only','no_dt_deviation'),default='full')
    p.add_argument('--task-size',type=float)
    p.add_argument('--device',default='auto')
    args=p.parse_args()
    c=load_config();c['environment']['mode']=args.mode
    c['environment']['force_task_size_mb']=args.task_size
    if args.algorithm=='maddpg':
        c['environment']['load_balance_penalty']=False
        c['environment']['use_digital_twin']=False
    env=RATOEnv(c)
    device=select_device(args.device)
    agent=make_agent(args.algorithm,env.observation_dim,env.M,env.K,c,device,args.seed)
    load_checkpoint(args.checkpoint,agent,device)
    events,summary=evaluate(agent,env,args.seed,args.episodes)
    folder=create_run_dir(ROOT,f'eval_{args.algorithm}_{args.mode}',args.seed)
    save_metadata(folder,c,ROOT)
    write_csv(folder/'raw_metrics'/'evaluation.csv',events)
    (folder/'summary'/'metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print('Saved:',folder)
if __name__=='__main__':main()
