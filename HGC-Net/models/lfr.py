import torch.nn as nn
from .scta import SCTA
from .grb import GRB


class LayerNorm2d(nn.Module):
    """LayerNorm over channels independently at each spatial location."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class LFRBlock(nn.Module):
    """Pre-norm residual attention followed by pre-norm residual GRB."""
    def __init__(self, dim, heads, expansion=2.66, scta_kernel=3,
                 normalize_qk=False, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.scta = SCTA(dim, heads, scta_kernel, bias)
        self.norm2 = LayerNorm2d(dim)
        self.grb = GRB(dim, expansion, bias)

    def forward(self, x):
        x = x + self.scta(self.norm1(x))
        x = x + self.grb(self.norm2(x))
        return x
