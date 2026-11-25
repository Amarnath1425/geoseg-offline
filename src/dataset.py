import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

BUILDING = 2   # LoveDA mask number for building
ROAD = 3       # LoveDA mask number for road
PATCH = 256    # crop size fed to the model
# ImageNet colour statistics, because the pretrained encoder expects them
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_pairs(root, scenes=("Urban",)):
    """Return a list of (image_path, mask_path) pairs."""
    pairs = []
    for scene in scenes:
        img_dir = os.path.join(root, scene, "images_png")
        mask_dir = os.path.join(root, scene, "masks_png")
        for name in sorted(os.listdir(img_dir)):
            if name.endswith(".png"):          # skips hidden files like .DS_Store
                pairs.append((os.path.join(img_dir, name),
                              os.path.join(mask_dir, name)))
    return pairs


def normalize(rgb):
    """uint8 HxWx3 array -> float32 tensor 3xHxW, normalised for the encoder."""
    x = rgb.astype(np.float32) / 255.0                 # colours 0..255 -> 0..1
    x = (x - MEAN) / STD                               # ImageNet normalisation
    x = np.ascontiguousarray(x.transpose(2, 0, 1))     # HWC -> CHW, clean memory layout
    return torch.from_numpy(x)


def remap_mask(mask):
    """LoveDA's 8 classes -> 0 other, 1 building, 2 road."""
    out = np.zeros(mask.shape, dtype=np.int64)         # int64 is what the loss needs
    out[mask == BUILDING] = 1
    out[mask == ROAD] = 2
    return out


class SegDataset(Dataset):
    def __init__(self, pairs, patch=PATCH, train=True, repeats=4):
        self.pairs = pairs
        self.patch = patch
        self.train = train
        # train: several random crops per image. val: a fixed 4x4 grid of 16 crops.
        self.repeats = repeats if train else 16

    def __len__(self):
        return len(self.pairs) * self.repeats

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx % len(self.pairs)]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        h, w = mask.shape
        if self.train:
            top = random.randint(0, h - self.patch)    # random crop corner
            left = random.randint(0, w - self.patch)
        else:
            k = idx // len(self.pairs)                 # which of the 16 grid cells
            top = (k // 4) * self.patch
            left = (k % 4) * self.patch
        img = img[top:top + self.patch, left:left + self.patch]
        mask = mask[top:top + self.patch, left:left + self.patch]
        if self.train:                                 # flips: same flip on photo AND mask
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1, :], mask[::-1, :]
        return normalize(img), torch.from_numpy(remap_mask(mask))