import torch
import numpy as np
import torch.nn as nn

def batchout_pre_hook(self, input):
    n_random = np.random.choice([0.3, -0.2, 0], p=[0.6, 0.3, 0.1])
    #n = 0.2
    # In case the model is in eval mode, don't perform batchout
    if not self.training:
        n = 0.0
    # Input would be tuple of size 1. To obtain the tensor, we need to access
    x = input[0]
    
    _r = np.linspace(x.shape[0]-1, 0, x.shape[0], dtype=int)

    # Obtain the feature randomly sampled from batch 
    _sample = x[_r]

    # Compute the direction of feature perturbation
    _d  = _sample - x
   
    # Augment and replace the 1st 'k' samples 
    x = x + (n * _d).detach()

    return x

