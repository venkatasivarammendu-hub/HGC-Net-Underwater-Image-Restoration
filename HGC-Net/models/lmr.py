import torch.nn as nn
from .lfr import LFRBlock


class LMRDecoder(nn.Module):
    """Latent Manifold Reconstruction decoder.

    Two reconstruction resolutions are used: half-resolution refinement and
    full-resolution refinement, with BFA skip fusion at each level.
    """
    def __init__(self, base_dim=48, heads=(1, 2, 4), depths=(3, 2),
                 expansion=2.66, scta_kernel=3):
        super().__init__()
        c1, c2, c3 = base_dim, base_dim * 2, base_dim * 4

        self.up_to_half = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.half_refine = nn.Sequential(*[
            LFRBlock(c2, heads[1], expansion, scta_kernel)
            for _ in range(depths[0])
        ])
        self.up_to_full = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.full_refine = nn.Sequential(*[
            LFRBlock(c1, heads[0], expansion, scta_kernel)
            for _ in range(depths[1])
        ])

    def forward(self, deep, skip_half, skip_full, bfa_half, bfa_full):
        x = self.up_to_half(deep)
        x = bfa_half(x, skip_half)
        x = self.half_refine(x)
        x = self.up_to_full(x)
        x = bfa_full(x, skip_full)
        x = self.full_refine(x)
        return x
