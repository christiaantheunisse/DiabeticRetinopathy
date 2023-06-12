import torch.nn as nn
import torch.nn.functional as F
from typing import List

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
    
class DeepDenseHead(nn.Module):
    """
        This head consists of three linear layers with a GELU activation for the hidden dimensions
        and a softmax activation applied at the output dimension.
    """
    def __init__(self, input_dims: int, output_dims: int, hidden_dims: List[int]=[700, 350]):
        super().__init__()
        assert len(hidden_dims) == 2, "There should be 2 input dimensions"
        
        self.linear1 = nn.Linear(input_dims, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], output_dims)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.linear3(x)
        x = F.softmax(x,dim=1)
        
        return x