"""Shared block layout in pixel-shuffle order."""
from torch import Tensor
from torch.nn import functional as F


def to_blocks(hr: Tensor, scale: int) -> Tensor:
    if hr.ndim != 4 or hr.shape[-2] % scale or hr.shape[-1] % scale:
        raise ValueError("Expected BCHW with spatial dimensions divisible by scale")
    b, c, h, w = hr.shape
    return F.pixel_unshuffle(hr, scale).reshape(b, c, scale**2, h // scale, w // scale)


def from_blocks(blocks: Tensor, scale: int) -> Tensor:
    b, c, q, h, w = blocks.shape
    if q != scale**2:
        raise ValueError("Block dimension must equal scale**2")
    return F.pixel_shuffle(blocks.reshape(b, c * q, h, w), scale)
