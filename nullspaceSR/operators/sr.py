import torch
from torch import Tensor
from torch.nn import functional as F

from .base import MeasurementOperator


class AveragePoolSR(MeasurementOperator):
    """Non-overlapping block-average downsampling and a right inverse.

    The image dimensions must be divisible by ``scale``. The pseudoinverse
    repeats each low-resolution value over its corresponding scale-by-scale
    block, so applying the block average recovers the measurement.
    """

    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
            raise ValueError("scale must be a positive integer")
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h % self.scale or w % self.scale:
            raise ValueError(f"Spatial dimensions {(h, w)} must be divisible by scale={self.scale}")
        return F.avg_pool2d(x, kernel_size=self.scale, stride=self.scale)

    def pseudoinverse(self, y: Tensor) -> Tensor:
        if y.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(y.shape)}")
        return y.repeat_interleave(self.scale, dim=-2).repeat_interleave(self.scale, dim=-1)
