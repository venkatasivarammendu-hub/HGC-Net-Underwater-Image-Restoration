import torch
from .hgcnet import HGCNet


def main():
    model = HGCNet(base_dim=24, encoder_depths=(1, 1, 1),
                   deep_semantic_depth=1, decoder_depths=(1, 1),
                   heads=(1, 2, 4), expansion=2.0)
    x = torch.rand(1, 3, 64, 64)
    y = model(x)
    assert y.shape == x.shape, (x.shape, y.shape)
    print("Smoke test passed:", tuple(y.shape))


if __name__ == "__main__":
    main()
