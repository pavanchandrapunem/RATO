#!/usr/bin/env python3
import time
import platform
import psutil
import torch

def main():
    print('Host:',platform.platform())
    print('Logical CPUs:',psutil.cpu_count(),'RAM GiB:',round(psutil.virtual_memory().total/2**30,2))
    print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a=torch.randn(512,512,device=device);b=torch.randn_like(a)
    start=time.perf_counter()
    for _ in range(100): _=a@b
    if device=='cuda':torch.cuda.synchronize()
    print('100 matrix multiplies, seconds:',round(time.perf_counter()-start,3))
if __name__=='__main__':main()
