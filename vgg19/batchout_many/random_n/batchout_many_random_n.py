import torch
import torch.nn as nn
import numpy as np

class Batchout_Many_Random(nn.Module):
    def __init__(self):
        super(Batchout_Many_Random, self).__init__()

    def _make_random(self, x, y):
        _r = np.random.randint(0, x.shape[0], x.shape[0])
        for i, r in enumerate(_r):
            if _r[i] == y[i]:
                while not _r[i] == y[i]:
                    _r[i] = np.random.randint(0, x.shape[0])

        #n_random = np.random.choice([0.3, -0.2], p=[0.6, 0.4])
        n_random = np.random.choice([0.3, -0.2], p=[0.6, 0.4])

        return _r, n_random

    # The difference with this class is we pass the random list and sometimes return it back 
    def forward(self, x, y, r=None, n=None):

        if r is None:
            # Sample random integers (with replacement) from 0 to m where m is batch size
            r, n = self._make_random(x, y)
    
        # Obtain the feature randomly sampled from batch 
        _sample = x[r]

        # Compute the direction of feature perturbation
        _d  = _sample - x
       
        # Augment and replace the 1st 'k' samples 
        x = x + (n * _d).detach()
        
        return x, r, n

# Run a test case to ensure class is working correctly
if __name__ == '__main__':

    a = Batchout_Many_Random()
    b = Batchout_Many_Random()
    b.eval()
    A = torch.rand([8, 4], requires_grad=True) # 20 is the batch size

    print(a.training)
    print(b.training)
    out, r, n = a.forward(A, np.random.randint(0, 8, 8))
    print(out)
    print(r)
    print(n)
