import torch
import torch.nn as nn
import numpy as np

class Batchout_Inference(nn.Module):
    def __init__(self, n):
        super(Batchout_Inference, self).__init__()
        self.n = 1 - n

    # The difference with this class is we pass the random list and sometimes return it back 
    def forward(self, x):

        #r = np.array([1, 0])
        r = np.zeros((x.shape[0],), dtype=int)

        # Obtain the feature randomly sampled from batch 
        _sample = x[r]

        # Compute the direction of feature perturbation
        _d  = _sample - x
       
        # Augment and replace the 1st 'k' samples 
        x = x + (self.n * _d).detach()
        
        return x


