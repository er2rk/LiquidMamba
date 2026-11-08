import torch.nn as nn
from ncps.torch import CfC
from mambapy.mamba import Mamba, MambaConfig


class LiquidNet(nn.Module):
    """
    CfC network from ncps.
    (B, L, F), timespans (B, L) -> (B, L, hidden_size)
    Currently: (64, 5, 5), (64, 5) -> (64, 5, 32)
    """
    def __init__(self, input_size, hidden_size, backbone_units=32):
        super().__init__()
        #For forward()'s timespans.
        self.hidden_size = hidden_size
        #Explicity setting backbone units to 32, keeping the block small. 
        self.cfc = CfC(input_size, hidden_size, backbone_units=backbone_units)

    def forward(self, x, timespans=None):
        if timespans is not None:
            #ncps multiplies each step's timespan with a (B, hidden_size) tensor: (B, L) -> (B, L, hidden_size)
            timespans = timespans.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        out, _ = self.cfc(x, timespans=timespans)
        return out


class MambaNet(nn.Module):
    """
    Mamba network from mambapy.
    (B, L, input_size) -> (B, L, hidden_size)
    Currently: (64, 63, 5) -> (64, 63, 32)
    Specifically not passing in timespans here. 
    """
    def __init__(self, input_size, hidden_size, n_layers=1):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.mamba = Mamba(MambaConfig(d_model=hidden_size, n_layers=n_layers))
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, x):
        return self.norm(self.mamba(self.proj(x)))