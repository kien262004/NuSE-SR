"""Đọc, ghi ảnh và tập ảnh valid."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..core.constants import IMAGE_EXT, S

def image_files(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)


def load_rgb(path):
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert("RGB"), dtype=np.uint8, copy=True)).permute(2, 0, 1)


def to_bmp(args):
    """(source, folder) -> cached BMP path; PNG 2K decoding is slow, BMP reads almost instantly."""
    source, folder = args
    target = Path(folder) / (Path(source).stem + ".bmp")
    if not target.exists():
        with Image.open(source) as image:
            image.convert("RGB").save(target.with_suffix(".partial"), format="BMP")
        target.with_suffix(".partial").rename(target)
    return str(target)


def save_png(path, image):
    """Clamp/round for viewing only; metrics never use this."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = image.detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(path)


class ValImages(Dataset):
    """Full images as uint8 (to keep RAM low), cropped bottom/right to a multiple of 4."""

    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        hr = load_rgb(self.files[index])
        h, w = hr.shape[-2] // S * S, hr.shape[-1] // S * S
        return hr[:, :h, :w].contiguous(), self.files[index].stem


def as_float(hr_uint8, device):
    return hr_uint8[None].to(device).float().div_(255)
