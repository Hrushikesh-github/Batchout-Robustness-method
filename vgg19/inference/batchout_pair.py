import torch
import torch.nn as nn
import numpy as np

class Batchout_Pair(nn.Module):
    def __init__(self, n):
        super(Batchout_Pair, self).__init__()
        self.n = n

    # The difference with this class is we pass the random list and sometimes return it back 
    def forward(self, x):

        r = np.array([1, 0])

        # Obtain the feature randomly sampled from batch 
        _sample = x[r]

        # Compute the direction of feature perturbation
        _d  = _sample - x
       
        # Augment and replace the 1st 'k' samples 
        x = x + (self.n * _d).detach()
        
        return x


# Run a test case to ensure class is working correctly
if __name__ == '__main__':

    b = Batchout_Pair(0.3)
    A = torch.rand([2, 3, 32, 32], requires_grad=True) # 20 is the batch size
    out = b(A)
    print(out.shape)
    print(A.shape)

