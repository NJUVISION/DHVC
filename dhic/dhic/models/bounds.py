import torch
import torch.nn as nn


class UpperBound(nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.register_buffer("bound", torch.tensor([float(bound)]))

    def forward(self, x):
        return torch.min(x, self.bound)
