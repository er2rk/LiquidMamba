import torch.nn as nn
from mambapy.mamba import MambaBlock, MambaConfig

from model.liquidnet import LiquidNet


class HybridBlock(nn.Module):
    """A CfC layer followed by a Mamba layer, each wrapped in a pre-norm residual."""

    def __init__(self, d_model, cfc_hidden):
        super().__init__()
        self.cfc_norm = nn.RMSNorm(d_model)
        self.cfc = LiquidNet(d_model, cfc_hidden, d_model)

        self.mamba_norm = nn.RMSNorm(d_model)
        self.mamba = MambaBlock(MambaConfig(d_model=d_model, n_layers=1))

    def forward(self, x, timespans=None):
        x = x + self.cfc(self.cfc_norm(x), timespans)
        x = x + self.mamba(self.mamba_norm(x))
        return x


class LiquidMamba(nn.Module):
    """Input projection, a stack of hybrid blocks, and a linear regression head."""

    def __init__(self, in_dim, d_model=32, n_layers=2, cfc_hidden=32):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([HybridBlock(d_model, cfc_hidden) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, timespans=None):
        """x: (B, L, in_dim), timespans: (B, L) or None -> (B, L, 1)"""
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, timespans)
        return self.head(self.norm(x))
