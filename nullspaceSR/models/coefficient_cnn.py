"""IMDN-style LR-domain predictor for NuSESR PCA coefficients."""
import torch
from torch import Tensor, nn


class ContrastAwareChannelAttention(nn.Module):
    """Channel attention from pooled mean and spatial contrast statistics."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        contrast = x.var(dim=(-2, -1), keepdim=True, unbiased=False).sqrt()
        weights = self.channel_gate(self.avg_pool(x) + contrast)
        return x * weights


class InformationMultiDistillationBlock(nn.Module):
    """Progressively distill feature channels, attend, then add a residual."""

    def __init__(self, channels: int, distillation_rate: float = 0.25) -> None:
        super().__init__()
        if not 0 < distillation_rate < 1:
            raise ValueError("distillation_rate must lie between 0 and 1")
        distilled = max(1, int(channels * distillation_rate))
        remaining = channels - distilled
        if remaining < 1:
            raise ValueError("IMDN blocks require at least two feature channels")

        self.distilled_channels = distilled
        self.remaining_channels = remaining
        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(remaining, channels, 3, padding=1)
        self.c3 = nn.Conv2d(remaining, channels, 3, padding=1)
        self.c4 = nn.Conv2d(remaining, distilled, 3, padding=1)
        self.cca = ContrastAwareChannelAttention(distilled * 4)
        self.fuse = nn.Conv2d(distilled * 4, channels, 1)

    def _distill(self, x: Tensor, conv: nn.Module) -> tuple[Tensor, Tensor]:
        features = self.act(conv(x))
        return features.split((self.distilled_channels, self.remaining_channels), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        d1, remaining = self._distill(x, self.c1)
        d2, remaining = self._distill(remaining, self.c2)
        d3, remaining = self._distill(remaining, self.c3)
        d4 = self.c4(remaining)
        distilled = self.cca(torch.cat((d1, d2, d3, d4), dim=1))
        return self.fuse(distilled) + x


class IMDN(nn.Module):
    """IMDN feature extractor with a NuSESR coefficient output head.

    The feature stem, six IMDBs, feature fusion, and LR residual follow IMDN.
    Only its RGB upsampling head is replaced: this head emits ``channels *
    rank`` coefficient maps at LR resolution for NuSESR's fixed decoder.
    """

    def __init__(self, channels: int = 3, rank: int = 8, width: int = 64) -> None:
        super().__init__()
        if min(channels, rank) < 1:
            raise ValueError("channels and rank must be positive")
        if width < 2:
            raise ValueError("width must be at least 2 for channel distillation")

        self.channels = channels
        self.rank = rank
        self.width = width
        self.num_modules = 6
        self.feature = nn.Conv2d(channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            InformationMultiDistillationBlock(width) for _ in range(self.num_modules)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(width * self.num_modules, width, 1),
            nn.LeakyReLU(0.05, inplace=True),
        )
        self.lr_conv = nn.Conv2d(width, width, 3, padding=1)
        self.coefficient_head = nn.Conv2d(width, channels * rank, 3, padding=1)

    def forward(self, lr: Tensor) -> Tensor:
        features = self.feature(lr)
        block_outputs = []
        x = features
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
        x = self.fusion(torch.cat(block_outputs, dim=1))
        x = self.lr_conv(x) + features
        return self.coefficient_head(x)


# Backward-compatible name for existing configs and user code.
SmallCoefficientCNN = IMDN
