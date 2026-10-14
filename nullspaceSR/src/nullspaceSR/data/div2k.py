"""DIV2K HR folders; LR is generated from the selected HR crop by A."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from .contracts import make_sample


class DIV2KDataset(Dataset):
    """RGB float [0,1]. No precomputed bicubic LR is used.

    Training uses reproducible random crops/augmentation indexed by epoch.
    Evaluation uses full images mod-cropped at bottom/right, batch size one.
    ``hr_dir`` must point directly to the desired split's HR image folder.
    """

    def __init__(self, hr_dir, *, scale=4, training=False, crop_size=256,
                 augment=True, repeat=1, seed=2026):
        self.root = Path(hr_dir).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"HR directory does not exist: {self.root}")
        if not isinstance(scale, int) or scale < 2:
            raise ValueError("scale must be an integer >= 2")
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError("repeat must be a positive integer")
        if training and (crop_size < scale or crop_size % scale):
            raise ValueError("Training crop_size must be positive and divisible by scale")
        self.files = sorted(p for p in self.root.iterdir()
                            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        if not self.files:
            raise ValueError(f"No HR images found directly in {self.root}")
        self.scale, self.training, self.crop_size = scale, training, crop_size
        self.augment, self.repeat, self.seed, self.epoch = augment, repeat if training else 1, seed, 0

    def __len__(self):
        return len(self.files) * self.repeat

    def set_epoch(self, epoch):
        self.epoch = epoch

    def manifest(self):
        return {"dataset": "hr_folder", "length": len(self),
                "sources": [str(p) for p in self.files], "scale": self.scale,
                "training": self.training, "crop_size": self.crop_size if self.training else None,
                "augment": self.augment if self.training else False,
                "repeat": self.repeat, "seed": self.seed}

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.training:
                rng = random.Random(self.seed + self.epoch * len(self) + index)
                w, h = image.size
                if min(w, h) < self.crop_size:
                    raise ValueError(f"{path.name}: image smaller than crop_size={self.crop_size}")
                left = rng.randrange(w - self.crop_size + 1)
                top = rng.randrange(h - self.crop_size + 1)
                image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                if self.augment:
                    if rng.random() < 0.5:
                        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    if rng.random() < 0.5:
                        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    turns = rng.randrange(4)
                    if turns:
                        image = image.transpose({1: Image.Transpose.ROTATE_90,
                                                 2: Image.Transpose.ROTATE_180,
                                                 3: Image.Transpose.ROTATE_270}[turns])
            else:
                w, h = image.size
                w, h = w - w % self.scale, h - h % self.scale
                if min(w, h) < self.scale:
                    raise ValueError(f"{path.name}: image too small for scale={self.scale}")
                image = image.crop((0, 0, w, h))
            w, h = image.size
            hr = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
            hr = hr.permute(2, 0, 1).contiguous().float().div_(255)
        lr = F.avg_pool2d(hr, kernel_size=self.scale, stride=self.scale)
        return make_sample(lr, hr, sample_id=path.name,
                           meta={"source": str(path), "scale": self.scale,
                                 "degradation": "average_pool_non_overlapping",
                                 "color_space": "RGB", "data_range": 1.0})
