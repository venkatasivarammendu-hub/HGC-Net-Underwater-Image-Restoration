import torch.nn as nn
from .lfr import LFRBlock


class HFCEncoder(nn.Module):
    """Hierarchical feature-compression encoder.

    Returns feature maps at full, half, and quarter resolution. The deep
    semantic core is represented by repeated LFR blocks at quarter resolution.
    """
    def __init__(self, base_dim=48, depths=(2, 3, 3), heads=(1, 2, 4),
                 deep_depth=3, expansion=2.66, scta_kernel=3,
                 normalize_qk=False):
        super().__init__()
        c1, c2, c3 = base_dim, base_dim * 2, base_dim * 4

        def stack(dim, head, depth):
            return nn.Sequential(*[
                LFRBlock(dim, head, expansion, scta_kernel)
                for _ in range(depth)
            ])

        self.ife = nn.Conv2d(3, c1, 3, padding=1)
        self.stage1 = stack(c1, heads[0], depths[0])
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.stage2 = stack(c2, heads[1], depths[1])
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.stage3 = stack(c3, heads[2], depths[2])
        self.deep_semantic_core = stack(c3, heads[2], deep_depth)

    def forward(self, x):
        f1 = self.stage1(self.ife(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        deep = self.deep_semantic_core(f3)
        return f1, f2, deep
