import torch.nn as nn
import torch.nn.functional as F


class GRB(nn.Module):
    """Gated Residual Block core: expand -> DWConv -> split -> GELU gate -> project.

    The residual addition is applied by LFRBlock, matching the paper's
    X_out = Conv1x1(GELU(X1) * X2) + X' formulation.
    """
    def __init__(self, dim, expansion=2.66, bias=False):
        super().__init__()
        hidden = max(2, int(dim * expansion))
        # The expansion tensor is split into two equal branches, so use 2*hidden.
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1,
            groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        y = self.dw(self.project_in(x))
        x1, x2 = y.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)
