"""NuSE-E residual predictor, independent of LR degradation and PCA synthesis."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RepConv(nn.Module):
    """Train with 3x3 + 1x1 + identity; fuse to one 3x3 for inference."""

    def __init__(self, width):
        super().__init__()
        self.conv3 = nn.Conv2d(width, width, 3, padding=1)
        self.conv1 = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return self.conv3(x) + self.conv1(x) + x

    @torch.no_grad()
    def fused(self):
        conv = nn.Conv2d(self.conv3.in_channels, self.conv3.out_channels, 3, padding=1)
        conv.to(device=self.conv3.weight.device, dtype=self.conv3.weight.dtype)
        weight = self.conv3.weight + F.pad(self.conv1.weight, (1, 1, 1, 1))
        index = torch.arange(weight.shape[0], device=weight.device)
        weight[index, index, 1, 1] += 1
        conv.weight.copy_(weight)
        conv.bias.copy_(self.conv3.bias + self.conv1.bias)
        return conv


class NuSEResBlock(nn.Module):
    def __init__(self, width, kind="plain"):
        super().__init__()
        if kind not in {"plain", "rep", "lk7"}:
            raise ValueError("block must be plain, rep or lk7")
        def conv():
            return RepConv(width) if kind == "rep" else nn.Conv2d(width, width, 3, padding=1)
        self.conv1, self.conv2 = conv(), conv()
        self.dw = nn.Conv2d(width, width, 7, padding=3, groups=width) if kind == "lk7" else None

    def forward(self, x):
        f = F.relu(self.conv1(x))
        if self.dw is not None:
            f = self.dw(f)
        return x + self.conv2(f)

    def fuse(self):
        for name in ("conv1", "conv2"):
            if isinstance(getattr(self, name), RepConv):
                setattr(self, name, getattr(self, name).fused())


def coefficient_layout(rank, eigenvalues, num_blocks):
    """Energy-based prefix budgets and stage depths, restricted to retained rank.

    Existing PCA files contain eigenvalues but no Wiener/R² statistics. Allocate
    by spectral energy (equivalent to the notebook rule with linear R² = 0).
    """
    if eigenvalues is None or not eigenvalues.numel():
        return [rank], [num_blocks]
    energy = eigenvalues[:, :rank].double().mean(0).clamp_min(0).tolist()
    total = sum(energy)
    if not total:
        return [rank], [num_blocks]
    cuts = [k for k in range(1, rank) if energy[k - 1] - energy[k] > 1e-6 * energy[k - 1]] + [rank]
    budgets = sorted({rank, *(next(k for k in cuts if sum(energy[:k]) >= (target - 1e-12) * total)
                              for target in (0.5, 0.8, 0.9, 1.0))})
    # Each stage needs at least one block; merge earliest budgets if necessary.
    while len(budgets) > num_blocks:
        budgets.pop(0)
    edges = [0, *budgets]
    shares = [num_blocks * sum(energy[a:b]) / total for a, b in zip(edges, edges[1:])]
    depths = [max(1, math.floor(share)) for share in shares]
    while sum(depths) < num_blocks:
        i = max(range(len(depths)), key=lambda i: shares[i] - depths[i])
        depths[i] += 1
    while sum(depths) > num_blocks:
        i = max((i for i in range(len(depths)) if depths[i] > 1), key=lambda i: depths[i] - shares[i])
        depths[i] -= 1
    return budgets, depths


class NuSEGroupHead(nn.Module):
    """Zero-initialized 1x1 readout, scaled to PCA coefficient units."""

    def __init__(self, width, channels, size, std):
        super().__init__()
        self.readout = nn.Conv2d(width, channels * size, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer("std", std.reshape(1, channels * size, 1, 1).float())

    def forward(self, x):
        return self.readout(x) * self.std


class NuSEBackbone(nn.Module):
    """NuSE-E stem/stages/group heads, or an image head without the PCA plugin.

    The predictor contains no measurement operator. ``k`` optionally selects a
    coefficient prefix; normal training and NuSESR use the complete rank.
    """

    def __init__(self, channels=3, rank=None, width=32, *, scale=4, num_blocks=7,
                 block="plain", budgets=None, stage_blocks=None, eigenvalues=None):
        super().__init__()
        for name, value in (("channels", channels), ("width", width), ("scale", scale),
                            ("num_blocks", num_blocks)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank < 1):
            raise ValueError("rank must be a positive integer")
        self.channels, self.rank, self.width, self.scale = channels, rank, width, scale
        self.num_blocks, self.block = num_blocks, block
        if rank is not None and eigenvalues is not None and eigenvalues.numel():
            if (eigenvalues.ndim != 2 or eigenvalues.shape[0] != channels or eigenvalues.shape[1] < rank
                    or not torch.isfinite(eigenvalues).all()):
                raise ValueError("eigenvalues must be finite with shape (channels, >=rank)")
        if rank is None:
            if budgets is not None or stage_blocks is not None:
                raise ValueError("budgets/stage_blocks only apply to coefficient mode")
            self.budgets, self.stage_blocks = [], [num_blocks]
        else:
            if budgets is None:
                budgets, derived_depths = coefficient_layout(rank, eigenvalues, num_blocks)
            else:
                derived_depths = None
            if (not budgets or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in budgets)
                    or list(budgets) != sorted(set(budgets)) or budgets[-1] != rank):
                raise ValueError("budgets must be strictly increasing positive integers ending at rank")
            if stage_blocks is None:
                if derived_depths is not None:
                    stage_blocks = derived_depths
                else:
                    if len(budgets) > num_blocks:
                        raise ValueError("num_blocks must provide at least one block per budget")
                    stage_blocks = [num_blocks // len(budgets) + (i < num_blocks % len(budgets))
                                    for i in range(len(budgets))]
            if (len(stage_blocks) != len(budgets)
                    or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in stage_blocks)
                    or sum(stage_blocks) != num_blocks):
                raise ValueError("stage_blocks must have one positive depth per budget and sum to num_blocks")
            self.budgets, self.stage_blocks = list(budgets), list(stage_blocks)
        self.stem = nn.Conv2d(channels, width, 3, padding=1)
        self.stages = nn.ModuleList(nn.Sequential(*(NuSEResBlock(width, block) for _ in range(n)))
                                    for n in self.stage_blocks)
        if rank is None:
            self.image_head = nn.Sequential(nn.Conv2d(width, channels * scale**2, 3, padding=1),
                                            nn.PixelShuffle(scale))
        else:
            std = torch.ones(channels, rank)
            if eigenvalues is not None and eigenvalues.numel():
                std = eigenvalues[:, :rank].clamp_min(1e-12).sqrt()
            self.groups = [b - a for a, b in zip([0, *self.budgets[:-1]], self.budgets)]
            self.heads = nn.ModuleList()
            start = 0
            for size in self.groups:
                self.heads.append(NuSEGroupHead(width, channels, size, std[:, start:start + size]))
                start += size

    def forward(self, lr, k=None):
        f = self.stem(lr)
        if self.rank is None:
            if k is not None:
                raise ValueError("k only applies to coefficient mode")
            for stage in self.stages:
                f = stage(f)
            return self.image_head(f)
        k = self.rank if k is None else k
        if k not in self.budgets:
            raise ValueError(f"k must be one of {self.budgets}")
        parts = []
        for stage, head, size in zip(self.stages[:self.budgets.index(k) + 1], self.heads, self.groups):
            f = stage(f)
            parts.append(head(f).reshape(lr.shape[0], self.channels, size, *lr.shape[-2:]))
        # Concatenate within each color channel, not group-major: decoder expects C*K.
        return torch.cat(parts, dim=2).flatten(1, 2)

    def fuse(self):
        """In-place inference conversion; call on a copy when retaining train checkpoints."""
        for stage in self.stages:
            for block in stage:
                block.fuse()
        return self


def build_predictor(backbone, channels, rank, width, *, scale=4, num_blocks=7,
                    block="plain", budgets=None, stage_blocks=None, decoder=None):
    if backbone == "imdn":
        from .coefficient_cnn import IMDN
        return IMDN(channels, rank, width, scale=scale)
    if backbone != "nuse":
        raise ValueError(f"Unknown backbone: {backbone}")
    return NuSEBackbone(channels, rank, width, scale=scale, num_blocks=num_blocks, block=block,
                        budgets=budgets, stage_blocks=stage_blocks,
                        eigenvalues=decoder.eigenvalues if decoder is not None else None)
