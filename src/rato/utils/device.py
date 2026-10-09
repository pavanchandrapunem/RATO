import torch

def select_device(requested='auto'):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(requested).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested, but torch.cuda.is_available() is false')
    return torch.device(requested)
