"""DIV2K HR folders; LR is generated from the selected HR crop by A."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .contracts import make_sample
from .degradation import make_lr, resolve_downsampler


class DIV2KDataset(Dataset):
    """Read HR only; generate floating RGB LR using the configured resize.

    Training uses reproducible random crops/augmentation indexed by epoch.
    Evaluation uses full images mod-cropped at bottom/right, batch size one.
    ``hr_dir`` must point directly to the desired split's HR image folder.
    ``resize.size`` sets LR height/width and the corresponding training HR crop.
    """

    def __init__(self, hr_dir, *, scale=4, training=False, crop_size=256,
                 augment=True, repeat=1, seed=2026, downsampler="average_pool", resize=None):
        self.root = Path(hr_dir).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"HR directory does not exist: {self.root}")
        if not isinstance(scale, int) or scale < 2:
            raise ValueError("scale must be an integer >= 2")
        if isinstance(scale, bool):
            raise ValueError("scale must be an integer >= 2")
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError("repeat must be a positive integer")
        if resize is not None and not isinstance(resize, dict):
            raise ValueError("resize must be a mapping")
        resize = dict(resize or {})
        unknown = set(resize) - {"method", "size", "antialias"}
        if unknown:
            raise ValueError(f"Unknown resize options: {sorted(unknown)}")
        self.lr_size = resize.get("size")
        if self.lr_size is not None:
            if (not isinstance(self.lr_size, (list, tuple)) or len(self.lr_size) != 2
                    or any(type(value) is not int or value < 1 for value in self.lr_size)):
                raise ValueError("resize.size must be null or [height, width] of positive integers")
            self.lr_size = tuple(self.lr_size)
        if training and self.lr_size is None and (type(crop_size) is not int
                or crop_size < scale or crop_size % scale):
            raise ValueError("Training crop_size must be positive and divisible by scale")
        self.antialias = resize.get("antialias", True)
        if not isinstance(self.antialias, bool):
            raise ValueError("resize.antialias must be a boolean")
        self.resize = {"method": resize.get("method", downsampler),
                       "size": list(self.lr_size) if self.lr_size else None,
                       "antialias": self.antialias}
        self.files = sorted(p for p in self.root.iterdir()
                            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        if not self.files:
            raise ValueError(f"No HR images found directly in {self.root}")
        self.scale, self.training, self.crop_size = scale, training, crop_size
        self.downsampler, self.downsampler_name = resolve_downsampler(self.resize["method"])
        self.augment, self.repeat, self.seed, self.epoch = augment, repeat if training else 1, seed, 0

    def __len__(self):
        return len(self.files) * self.repeat

    def set_epoch(self, epoch):
        self.epoch = epoch

    def manifest(self):
        return {"dataset": "hr_folder", "length": len(self),
                "sources": [str(p) for p in self.files], "scale": self.scale,
                "training": self.training,
                "crop_size": (list(value * self.scale for value in self.lr_size)
                              if self.lr_size else self.crop_size) if self.training else None,
                "augment": self.augment if self.training else False,
                "repeat": self.repeat, "seed": self.seed,
                "downsampler": self.downsampler_name, "resize": self.resize}

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.training:
                rng = random.Random(self.seed + self.epoch * len(self) + index)
                w, h = image.size
                crop_h, crop_w = (tuple(value * self.scale for value in self.lr_size)
                                  if self.lr_size else (self.crop_size, self.crop_size))
                if w < crop_w or h < crop_h:
                    raise ValueError(f"{path.name}: image smaller than HR crop {(crop_h, crop_w)}")
                left = rng.randrange(w - crop_w + 1)
                top = rng.randrange(h - crop_h + 1)
                image = image.crop((left, top, left + crop_w, top + crop_h))
                if self.augment:
                    if rng.random() < 0.5:
                        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    if rng.random() < 0.5:
                        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    turns = rng.randrange(4) if crop_h == crop_w else rng.choice((0, 2))
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
        if self.lr_size and (h, w) != tuple(value * self.scale for value in self.lr_size):
            raise ValueError("resize.size must match HR dimensions / model scale; "
                             "use size: null for full-image validation/test")
        lr = make_lr(hr, scale=self.scale, downsampler=self.downsampler,
                     size=self.lr_size, antialias=self.antialias)
        return make_sample(lr, hr, sample_id=path.name,
                           meta={"source": str(path), "scale": self.scale,
                                 "degradation": self.downsampler_name,
                                 "resize": self.resize,
                                 "color_space": "RGB", "data_range": 1.0})
