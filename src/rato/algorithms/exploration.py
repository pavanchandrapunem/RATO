import numpy as np

def continuous_action(actor, observations, rng, epsilon, noise_std, device):
    import torch
    with torch.no_grad():
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        act = actor(obs).cpu().numpy()
    for m in range(len(act)):
        if rng.random() < epsilon:
            act[m] += rng.normal(0, noise_std, size=act.shape[1])
    return act.astype(np.float32)
