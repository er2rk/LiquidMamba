#import torch
import torch.nn as nn
from ncps.torch import CfC

class LiquidNet(nn.Module):
    """
    CfC network from ncps.
    (B, L, input_size), timespans (B, L) -> (B, L, output_size)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cfc = CfC(input_size, hidden_size, proj_size=output_size)

    def forward(self, x, timespans=None):
        out, _ = self.cfc(x, timespans=timespans)
        return out