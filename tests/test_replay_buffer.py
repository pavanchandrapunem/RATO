import numpy as np
import torch
from rato.algorithms.replay_buffer import ReplayBuffer

def test_replay_keeps_joint_states():
    b=ReplayBuffer(2,1)
    for i in range(3):
        b.add(np.full((3,8),i),np.zeros((3,7)),np.ones(3),np.ones((3,8)),False)
    assert len(b)==2
    tensors=b.sample(2,torch.device('cpu'))
    assert tensors[0].shape==(2,3,8)
    assert not np.any(tensors[0].numpy()==0)
