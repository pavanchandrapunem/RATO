#!/usr/bin/env python3
"""Reproducible experiment sweeps. Never fills plots with paper-image datapoints."""
import argparse
import json
from pathlib import Path
import numpy as np
from rato.utils.config import ROOT,load_config
from rato.utils.device import select_device
from rato.utils.logging_utils import create_run_dir,save_metadata,write_csv
from rato.environment.rato_env import RATOEnv
from rato.algorithms import make_agent,ALGORITHMS
from rato.utils.checkpoint import load_checkpoint
from rato.communication.noma import noma_rates,power_allocation
from rato.communication.oma import oma_rates

def throughput_sweep(cfg,seed):
    rng=np.random.default_rng(seed)
    P=cfg['paper']['communication']
    K=cfg['paper']['network']['num_edge_servers']
    M=cfg['paper']['network']['num_ues']
    gains=10**(rng.uniform(P['channel_gain_db_min'],P['channel_gain_db_max'],size=M)/10)
    assoc=np.arange(M)%K
    rows=[]
    for factor,values in [('bandwidth_mhz',[10,15,20,25,30]),
                          ('power_w',[10,15,20,25,30]),
                          ('num_edges',[3,4,5,6,7]),
                          ('ues_per_edge',[2,3,4,5,6])]:
        for value in values:
            nk=int(value) if factor=='num_edges' else K
            nm=nk*int(value) if factor=='ues_per_edge' else M
            g=np.resize(gains,nm)
            ix=np.arange(nm)%nk
            bandwidth=float(value)*1e6 if factor=='bandwidth_mhz' else P['bandwidth_hz']
            total_power=float(value) if factor=='power_w' else P['total_power_w']
            cluster_p=power_allocation(nk,total_power)
            for access,fn in [('noma',noma_rates),('oma',oma_rates)]:
                rate=fn(g,ix,cluster_p,bandwidth,P['noise_power_w'])
                rows.append(dict(experiment=factor,x=value,access=access,
                                 throughput_mbps=rate.sum()/1e6,seed=seed))
    for power in [10,15,20,25,30]:
        for scheme in ('equal','linear','exponential'):
            cluster_p=power_allocation(K,power,scheme)
            rate=noma_rates(gains,assoc,cluster_p,P['bandwidth_hz'],P['noise_power_w'])
            rows.append(dict(experiment='power_allocation',x=power,access=scheme,
                             throughput_mbps=rate.sum()/1e6,seed=seed))
    return rows

def parse_checkpoints(values):
    result={}
    for spec in values:
        if '=' not in spec:raise ValueError('Use name=/absolute/path/latest.pt')
        name,path=spec.split('=',1)
        if name not in ALGORITHMS:raise ValueError(name)
        result[name]=Path(path)
    return result

def evaluate_policy(cfg, name, path, seed, episodes, device):
    from evaluate import evaluate
    env=RATOEnv(cfg)
    agent=make_agent(name,env.observation_dim,env.M,env.K,cfg,device,seed)
    load_checkpoint(path,agent,device)
    return evaluate(agent,env,seed,episodes)[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['throughput','algorithms','schemes','task_sweep'],default='throughput')
    p.add_argument('--checkpoints',nargs='*',default=[],help='algorithm=/path/to/latest.pt')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--device',default='auto')
    args=p.parse_args()
    c=load_config()
    folder=create_run_dir(ROOT,args.kind,args.seed)
    save_metadata(folder,c,ROOT)
    if args.kind=='throughput':
        rows=throughput_sweep(c,args.seed)
    else:
        cps=parse_checkpoints(args.checkpoints)
        if not cps:raise SystemExit('Trained checkpoints required; no fake learned-policy results are generated.')
        dev=select_device(args.device)
        rows=[]
        modes={'schemes':['full','edge_only','cloud_only','no_dt_deviation'],
               'algorithms':['full'],'task_sweep':['full']}[args.kind]
        sizes=[30,40,50,60] if args.kind=='task_sweep' else [None]
        for name,path in cps.items():
            for mode in modes:
                for size in sizes:
                    cc=load_config();cc['environment']['mode']=mode
                    cc['environment']['force_task_size_mb']=size
                    if name=='maddpg':
                        cc['environment']['use_digital_twin']=False
                        cc['environment']['load_balance_penalty']=False
                    summary=evaluate_policy(cc,name,path,args.seed,args.episodes,dev)
                    rows.append(dict(algorithm=name,mode=mode,task_size_mb=size or 0,seed=args.seed,**summary))
                    print(name,mode,size,summary['reward'],flush=True)
    write_csv(folder/'raw_metrics'/(args.kind+'.csv'),rows)
    (folder/'summary'/'experiment.json').write_text(json.dumps(dict(kind=args.kind,seed=args.seed,count=len(rows)),indent=2))
    print('Saved:',folder)
if __name__=='__main__':main()
