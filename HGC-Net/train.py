import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from models import HGCNet
from losses import CharbonnierLoss, EdgeLoss, FFTLoss


class PairedImages(Dataset):
    def __init__(self, input_dir, target_dir, patch_size=None, train=False):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.train = train
        valid = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        inputs = {p.stem: p for p in self.input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in valid}
        targets = {p.stem: p for p in self.target_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in valid}
        if inputs.keys() != targets.keys():
            raise ValueError("Input and target directories must contain matching filename stems.")
        self.names = sorted(inputs)
        self.inputs, self.targets = inputs, targets

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        inp = Image.open(self.inputs[name]).convert("RGB")
        tgt = Image.open(self.targets[name]).convert("RGB")
        if inp.size != tgt.size:
            raise ValueError(f"Image size mismatch for pair {name}")
        if self.train and self.patch_size:
            w, h = inp.size
            p = self.patch_size
            if w < p or h < p:
                pad_w, pad_h = max(0, p-w), max(0, p-h)
                inp = TF.pad(inp, [0, 0, pad_w, pad_h], padding_mode="reflect")
                tgt = TF.pad(tgt, [0, 0, pad_w, pad_h], padding_mode="reflect")
                w, h = inp.size
            top = random.randint(0, h-p)
            left = random.randint(0, w-p)
            inp, tgt = TF.crop(inp, top, left, p, p), TF.crop(tgt, top, left, p, p)
            if random.random() < 0.5:
                inp, tgt = TF.hflip(inp), TF.hflip(tgt)
        return TF.to_tensor(inp), TF.to_tensor(tgt), name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_hgcnet.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    seed = int(cfg.get("seed", 2026))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    m = cfg["model"]
    t = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HGCNet(
        in_channels=m.get("in_channels", 3),
        base_dim=m["base_dim"],
        encoder_depths=tuple(m["encoder_depths"]),
        deep_semantic_depth=m["deep_semantic_depth"],
        decoder_depths=tuple(m["decoder_depths"]),
        heads=tuple(m["heads"]),
        expansion=m["expansion"],
        scta_kernel=m["scta_kernel"],
        normalize_qk=m.get("normalize_qk", False),
    ).to(device)

    train_ds = PairedImages(t["train_input_dir"], t["train_target_dir"],
                            t["patch_size"], train=True)
    val_ds = PairedImages(t["val_input_dir"], t["val_target_dir"], train=False)
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                          num_workers=t["num_workers"], pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"],
                                  weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    char_loss, edge_loss, fft_loss = CharbonnierLoss(), EdgeLoss().to(device), FFTLoss()
    out_dir = Path(t["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(t["epochs"]):
        model.train()
        train_sum = 0.0
        for inp, target, _ in train_dl:
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = (char_loss(pred, target)
                    + t["edge_weight"] * edge_loss(pred, target)
                    + t["fft_weight"] * fft_loss(pred, target))
            loss.backward()
            optimizer.step()
            train_sum += loss.item()
        scheduler.step()

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for inp, target, _ in val_dl:
                inp, target = inp.to(device), target.to(device)
                pred = model(inp)
                loss = (char_loss(pred, target)
                        + t["edge_weight"] * edge_loss(pred, target)
                        + t["fft_weight"] * fft_loss(pred, target))
                val_sum += loss.item()
        train_avg = train_sum / max(1, len(train_dl))
        val_avg = val_sum / max(1, len(val_dl))
        print(f"Epoch {epoch+1:03d}/{t['epochs']} train={train_avg:.6f} val={val_avg:.6f}")

        state = {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "config": cfg,
                 "val_loss": val_avg}
        torch.save(state, out_dir / "last.pth")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(state, out_dir / "best.pth")


if __name__ == "__main__":
    main()
