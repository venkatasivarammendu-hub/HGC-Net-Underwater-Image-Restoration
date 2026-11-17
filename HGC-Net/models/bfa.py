import torch.nn as nn
import torch.nn.functional as F


class BFA(nn.Module):

    def __init__(self, decoder_channels, skip_channels):
        super().__init__()
        self.decoder_proj = nn.Conv2d(decoder_channels, skip_channels, 1)
        self.skip_proj = nn.Conv2d(skip_channels, skip_channels, 1)

    def forward(self, decoder, skip):
        decoder = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        decoder = self.decoder_proj(decoder)
        skip = self.skip_proj(skip)
        return decoder + skip
