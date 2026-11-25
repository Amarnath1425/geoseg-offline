import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   # lets unsupported Apple-GPU ops run on CPU
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from src.dataset import list_pairs, SegDataset
from src.model import build_model, get_device

ROOT = "data/Val"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BATCH = 16            # lower to 8 if you get an out-of-memory error


@torch.no_grad()
def evaluate(model, loader, device):
    """IoU per class = overlap / union, over the whole held-out set."""
    model.eval()
    inter, union = np.zeros(3), np.zeros(3)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)                 # most likely class per pixel
        for c in range(3):
            p, t = pred == c, y == c
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()
    return inter / np.maximum(union, 1)


def main():
    device = get_device()
    print("device:", device)
    pairs = list_pairs(ROOT, scenes=("Urban",))
    train_pairs, val_pairs = pairs[:-70], pairs[-70:]  # last 70 images are held out
    print(len(train_pairs), "train images,", len(val_pairs), "held-out images")
    train_loader = DataLoader(SegDataset(train_pairs, train=True), batch_size=BATCH,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(SegDataset(val_pairs, train=False), batch_size=BATCH,
                            shuffle=False, num_workers=0)

    model = build_model(pretrained=True).to(device)
    # Roads and buildings are rare pixels, so errors on them count 3x
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 3.0], device=device))
    dice = smp.losses.DiceLoss(mode="multiclass")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    os.makedirs("models", exist_ok=True)
    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start, total = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()                     # clear last step's gradients
            logits = model(x)
            loss = ce(logits, y) + dice(logits, y)
            loss.backward()                           # compute gradients
            optimizer.step()                          # nudge the weights
            total += loss.item()
        ious = evaluate(model, val_loader, device)
        score = (ious[1] + ious[2]) / 2               # mean of building and road IoU
        print(f"epoch {epoch}/{EPOCHS} loss {total/len(train_loader):.3f} "
              f"IoU other {ious[0]:.3f} building {ious[1]:.3f} road {ious[2]:.3f} "
              f"({time.time()-start:.0f}s)")
        if score > best:
            best = score
            torch.save(model.state_dict(), "models/best.pth")
            print("  saved models/best.pth")


if __name__ == "__main__":
    main()