#!/usr/bin/env python3
import platform
import sys
import torch
import numpy
import yaml

def main():
    print('Python:', sys.version.split()[0], '|', platform.platform())
    print('NumPy:',numpy.__version__,'PyYAML:',yaml.__version__)
    print('PyTorch:',torch.__version__, '| compiled CUDA:',torch.version.cuda)
    print('CUDA available:',torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU:',torch.cuda.get_device_name(0))
        print('GPU memory GiB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,2))
        x = torch.randn(64,64,device=device); y = x @ x
        torch.cuda.synchronize()
        print('CUDA matrix test:', tuple(y.shape), 'PASS')
    else:
        print('CPU-only training available')
if __name__ == '__main__': main()
