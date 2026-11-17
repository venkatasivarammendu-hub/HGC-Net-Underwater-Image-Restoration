# HGC-Net reference implementation scaffold

This repository separates the HGC-Net components into modules for readability and reproducibility.



## Architecture

- `models/scta.py`: Spatially Constrained Transposed Attention (SCTA)
- `models/grb.py`: Gated Residual Block (GRB)
- `models/lfr.py`: pre-normalized residual SCTA + GRB block
- `models/hfc.py`: hierarchical encoder / deep semantic compression stages
- `models/bfa.py`: decoder feature alignment and fusion
- `models/lmr.py`: reconstruction-side feature refinement
- `models/hgcnet.py`: complete image-to-image network
- `losses.py`: Charbonnier, Laplacian-style edge, and FFT losses
- `train.py`: minimal paired-image training script

## Environment

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Dataset layout

The training script expects paired images with matching filename stems:

```text
data/
  trainA/   # degraded underwater images
  trainB/   # reference/target images
  valA/
  valB/
```

Edit paths in `configs/paper_hgcnet.yaml`, then run:

```bash
python train.py --config configs/paper_hgcnet.yaml
```

The script saves checkpoints and training logs under the configured output directory.

