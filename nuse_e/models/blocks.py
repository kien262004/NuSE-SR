"""Khối của trunk: RepConv và residual block."""
import torch
from torch import nn
from torch.nn import functional as F

class RepConv(nn.Module):
    """Conv3x3 trained as 3x3 + 1x1 + identity (structural re-parameterisation, RepVGG / ECB style);
    `fuse()` folds the three branches into one Conv3x3, so inference cost equals a plain Conv3x3."""

    def __init__(self, width):
        super().__init__()
        self.conv3 = nn.Conv2d(width, width, 3, padding=1)
        self.conv1 = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return self.conv3(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        width = self.conv3.in_channels
        conv = nn.Conv2d(width, width, 3, padding=1).to(self.conv3.weight.device)
        weight = self.conv3.weight + F.pad(self.conv1.weight, [1, 1, 1, 1])
        weight[torch.arange(width), torch.arange(width), 1, 1] += 1.0
        conv.weight.copy_(weight)
        conv.bias.copy_(self.conv3.bias + self.conv1.bias)
        return conv


class ResBlock(nn.Module):
    """Generic residual block (ResNet / EDSR): Conv3x3 -> ReLU -> Conv3x3, identity skip.

    kind = "plain": as above. "rep": both convs are RepConv while training (same inference block).
    "lk7": Conv3x3 -> ReLU -> depthwise Conv7x7 -> Conv3x3 (larger receptive field for +50 params/channel).
    """

    def __init__(self, width, kind="plain"):
        super().__init__()
        conv = RepConv if kind == "rep" else (lambda w: nn.Conv2d(w, w, 3, padding=1))
        self.conv1 = conv(width)
        self.conv2 = conv(width)
        self.dw = nn.Conv2d(width, width, 7, padding=3, groups=width) if kind == "lk7" else None

    def forward(self, x):
        f = F.relu(self.conv1(x))
        if self.dw is not None:
            f = self.dw(f)
        return x + self.conv2(f)

    def fuse(self):
        for name in ("conv1", "conv2"):
            if isinstance(getattr(self, name), RepConv):
                setattr(self, name, getattr(self, name).fuse())

    @staticmethod
    def inference_params(width, kind):
        return 2 * (9 * width * width + width) + (49 * width + width if kind == "lk7" else 0)
