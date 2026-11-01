"""Conv7 baseline, selective local correction and optional global gating."""
import torch
from torch import nn
from torch.nn import functional as F


class AblationBackbone(nn.Module):
    """Coefficient predictor; the caller installs the fixed PCA decoder.

    Variants: linear, cnn, local, global, wider, context. Only global contains
    image-wide pooling. `wider` and `context` provide local cost controls.
    """
    VARIANTS = {"linear", "cnn", "local", "global", "wider", "context"}

    def __init__(self, channels=3, rank=8, scale=4, *, variant="local", width=16,
                 corrected=2, linear_groups="channel", gate_hidden=8):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown ablation variant {variant}")
        if any(type(x) is not int or x < 1 for x in (channels, rank, scale, width, corrected, gate_hidden)):
            raise ValueError("Architecture sizes must be positive integers")
        if corrected > rank or linear_groups not in {"channel", "rgb"}:
            raise ValueError("corrected must be <= rank and linear_groups must be channel or rgb")
        self.channels, self.rank, self.scale = channels, rank, scale
        self.variant, self.width, self.corrected = variant, width, corrected
        self.linear_groups, self.gate_hidden = linear_groups, gate_hidden
        self.linear = None if variant == "cnn" else nn.Conv2d(
            channels, channels * rank, 7, padding=3, padding_mode="replicate",
            groups=channels if linear_groups == "channel" else 1)
        if variant != "linear":
            dilation = 2 if variant == "context" else 1
            self.stem = nn.Conv2d(channels, width, 3, padding=1, padding_mode="replicate")
            self.local = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation,
                                   padding_mode="replicate")
            size = rank if variant == "cnn" else corrected
            self.head = nn.Conv2d(width, channels * size, 1)
            if variant != "cnn":
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            if variant == "global":
                self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, gate_hidden, 1),
                                          nn.ReLU(), nn.Conv2d(gate_hidden, width, 1), nn.Tanh())
                self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, lr):
        b, _, h, w = lr.shape
        base = self.linear(lr) if self.linear is not None else None
        if self.variant == "linear":
            return base
        features = F.relu(self.local(F.relu(self.stem(lr))))
        if self.variant == "global":
            features = features * (1 + self.gamma * self.gate(features))
        delta = self.head(features)
        if self.variant == "cnn":
            return delta
        delta = delta.reshape(b, self.channels, self.corrected, h, w)
        if self.corrected < self.rank:
            delta = torch.cat((delta, delta.new_zeros(b, self.channels, self.rank - self.corrected, h, w)), 2)
        return base + delta.flatten(1, 2)
