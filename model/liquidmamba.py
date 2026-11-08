import torch
import torch.nn as nn
from model.blocks import LiquidNet, MambaNet


class LiquidMamba(nn.Module):
    """
    Parallel hybrid. Mamba reads the full window, CfC reads the last short_window days.
    A gate blends their last day's outputs, then a linear head predicts one return.
    (B, L, F), timespans (B, L) -> (B,)
    """
    def __init__(self, input_size, hidden_size=32, backbone_units=32, n_layers=1, short_window=5):
        super().__init__()
        self.short_window = short_window
        self.mamba = MambaNet(input_size, hidden_size, n_layers)
        self.cfc = LiquidNet(input_size, hidden_size, backbone_units)
        self.gate = nn.Linear(2 * hidden_size, 1)
        self.head = nn.Linear(hidden_size, 1)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, timespans):
        #Long branch, full window. Last day's output summarises the quarter.
        long = self.mamba(x)[:, -1]
        #Short branch, summarizes the week.
        short = self.cfc(x[:, -self.short_window:], timespans[:, -self.short_window:])[:, -1]

        g = torch.sigmoid(self.gate(torch.cat([long, short], dim=-1)))

        return self.head(g * long + (1 - g) * short).squeeze(-1)