import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


class EdgeLoss(nn.Module):
    """Laplacian-pyramid-style edge loss inspired by the paper description."""
    def __init__(self):
        super().__init__()
        k = torch.tensor([0.05, 0.25, 0.40, 0.25, 0.05], dtype=torch.float32)
        kernel2d = torch.outer(k, k).view(1, 1, 5, 5)
        self.register_buffer("kernel", kernel2d.repeat(3, 1, 1, 1))
        self.char = CharbonnierLoss()

    def blur(self, image):
        return F.conv2d(
            F.pad(image, (2, 2, 2, 2), mode="replicate"),
            self.kernel, groups=image.shape[1]
        )

    def laplacian(self, image):
        blurred = self.blur(image)
        down = blurred[:, :, ::2, ::2]
        up = torch.zeros_like(blurred)
        up[:, :, ::2, ::2] = down * 4.0
        return image - self.blur(up)

    def forward(self, pred, target):
        return self.char(self.laplacian(pred), self.laplacian(target))


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        return torch.abs(torch.fft.fft2(pred) - torch.fft.fft2(target)).mean()


def restoration_loss(pred, target, edge_weight=0.05, fft_weight=0.01):
    char = CharbonnierLoss()(pred, target)
    edge = EdgeLoss().to(pred.device)(pred, target)
    fft = FFTLoss()(pred, target)
    return char + edge_weight * edge + fft_weight * fft
