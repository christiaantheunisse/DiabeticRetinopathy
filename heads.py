import torch.nn as nn
import torch.nn.functional as F

"""
    The classification heads are defined in this file. A head should be a subclass
    of nn.module and should at least implement the `__init__()` and `forward()` function.
"""


class PassThrough(nn.Module):
    """
        This is a dummy head that just passes through the encoder output
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    
class OneLayer(nn.Module):
    """
        This head consists of one linear layer with a softmax activation applied at the output.
    """
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        x = self.linear(x)
        x = F.softmax(x,dim=1)
        
        return x