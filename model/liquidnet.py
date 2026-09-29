#import torch
import torch.nn as nn
from ncps.torch import CfC


"""class LiquidTimeStep(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeStep, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x, h):
        dx = torch.tanh(self.W_in(x) + self.W_h(h))
        h_new = h + (dx - h) / self.tau
        return h_new"""
    

class LiquidNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cfc = CfC(input_size, hidden_size, proj_size=output_size)

    def forward(self, x):
        out, _ = self.cfc(x)
        return out