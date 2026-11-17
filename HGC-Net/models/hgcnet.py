import torch
import torch.nn as nn
from .hfc import HFCEncoder
from .lmr import LMRDecoder
from .bfa import BFA


class HGCNet(nn.Module):
   
    def __init__(self, in_channels=3, base_dim=48, encoder_depths=(2, 3, 3),
                 deep_semantic_depth=3, decoder_depths=(3, 2),
                 heads=(1, 2, 4), expansion=2.66, scta_kernel=3,
                 normalize_qk=True):
        super().__init__()
        self.encoder = HFCEncoder(
            base_dim=base_dim, depths=encoder_depths, heads=heads,
            deep_depth=deep_semantic_depth, expansion=expansion,
            scta_kernel=scta_kernel
        )
        self.bfa_half = BFA(base_dim * 2, base_dim * 2)
        self.bfa_full = BFA(base_dim, base_dim)
        self.decoder = LMRDecoder(
            base_dim=base_dim, heads=heads, depths=decoder_depths,
            expansion=expansion, scta_kernel=scta_kernel,
            normalize_qk=normalize_qk
        )
        self.iru = nn.Conv2d(base_dim, in_channels, 3, padding=1)

    def forward(self, image):
        f1, f2, deep = self.encoder(image)
        features = self.decoder(
            deep, f2, f1, self.bfa_half, self.bfa_full
        )
        return self.iru(features) + image
