import torch

def save_checkpoint(path, agent, episode, epsilon, extra=None):
    state = {'episode': episode, 'epsilon': epsilon,
             'agent': agent.state_dict(), 'extra': extra or {}}
    torch.save(state, path)

def load_checkpoint(path, agent, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(state['agent'])
    return state
