"""Bộ lấy crop HR ngẫu nhiên để train (crop, lật, xoay, lặp)."""
import math
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class TrainCrops(Dataset):
    """HR crops (default 256 = LR 64 x 4); LR is generated on the GPU by `degrade`.

    Same sampling as the repo DIV2KDataset: per-item RNG seeded by (seed, epoch, index),
    flips + 90-degree rotations, `repeat` crops per image per epoch.
    """

    def __init__(self, files, crop=256, repeat=20, augment=True, seed=2026):
        self.files, self.crop, self.repeat, self.augment, self.seed = list(files), crop, repeat, augment, seed
        self.epoch = 0

    def __len__(self):
        return len(self.files) * self.repeat

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        with Image.open(path) as image:
            image = image.convert("RGB")
            w, h = image.size
            left, top = rng.randrange(w - self.crop + 1), rng.randrange(h - self.crop + 1)
            image = image.crop((left, top, left + self.crop, top + self.crop))
            if self.augment:
                if rng.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if rng.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                turns = rng.randrange(4)
                if turns:
                    image = image.transpose({1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                                             3: Image.Transpose.ROTATE_270}[turns])
            hr = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1)
        return hr.float().div_(255), index % len(self.files)


class RamCropLoader:
    """Same sampling as TrainCrops (per-item RNG seeded by (seed, epoch, index), flips + rotations,
    `repeat` crops per image per epoch); yields uint8 batches (B,3,crop,crop) straight from RAM."""

    def __init__(self, images, batch, seed, crop=256, repeat=20, augment=True):
        self.images, self.batch, self.seed, self.crop, self.repeat, self.augment = images, batch, seed, crop, repeat, augment
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.images) * self.repeat / self.batch)

    def item(self, index):
        i = index % len(self.images)
        rng = random.Random(self.seed + self.epoch * len(self.images) * self.repeat + index)
        h, w = self.images.size(i)
        left, top = rng.randrange(w - self.crop + 1), rng.randrange(h - self.crop + 1)
        hr = self.images.crop(i, top, left, self.crop)
        if self.augment:
            if rng.random() < 0.5:
                hr = hr.flip(-1)
            if rng.random() < 0.5:
                hr = hr.flip(-2)
            turns = rng.randrange(4)
            if turns:
                hr = torch.rot90(hr, turns, (-2, -1))
        return hr.contiguous(), i

    def __iter__(self):
        n = len(self.images) * self.repeat
        order = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for start in range(0, n, self.batch):
            items = [self.item(j) for j in order[start:start + self.batch]]
            yield torch.stack([hr for hr, _ in items]), torch.tensor([i for _, i in items])


def train_loader(files, *, batch, workers, seed, crop=256, repeat=20, augment=True):
    data = TrainCrops(files, crop, repeat, augment, seed)
    loader = DataLoader(data, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True,
                        drop_last=False, persistent_workers=workers > 0,
                        generator=torch.Generator().manual_seed(seed))
    return data, loader
