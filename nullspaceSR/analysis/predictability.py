"""Streaming ridge statistics and controlled-receptive-field CNN probes."""
import torch
from torch import nn
from torch.nn import functional as F


def sample_indices(batch, height, width, samples, generator):
    """Same deterministic spatial sample protocol for every predictor/context."""
    n = height * width
    return torch.stack([torch.randperm(n, generator=generator)[:min(samples, n)]
                        for _ in range(batch)])


def sampled_values(tensor, indices):
    """B,C,H,W -> sampled B*N,C, without changing channel/component order."""
    flat = tensor.flatten(2).transpose(1, 2)
    return flat.gather(1, indices.to(tensor.device)[..., None].expand(-1, -1, tensor.shape[1])).flatten(0, 1)


def sampled_windows(lr, context, indices, channel_mode):
    """Gather only sampled windows; never unfold a complete DIV2K image.

    Returns (1,N,3*r*r) for joint RGB or (3,N,r*r) for same-channel fits.
    Padding and flattening match Conv2d with replicate padding exactly.
    """
    if context < 1 or context % 2 != 1:
        raise ValueError("context must be a positive odd integer")
    if channel_mode not in {"rgb", "same_channel"}:
        raise ValueError("channel_mode must be rgb or same_channel")
    b, c, h, w = lr.shape
    padded = F.pad(lr, (context // 2,) * 4, mode="replicate")
    indices = indices.to(lr.device)
    dy, dx = torch.meshgrid(torch.arange(context, device=lr.device),
                            torch.arange(context, device=lr.device), indexing="ij")
    yy = indices[..., None] // w + dy.flatten()
    xx = indices[..., None] % w + dx.flatten()
    locations = yy * padded.shape[-1] + xx
    windows = padded.flatten(2).gather(2, locations.flatten(1)[:, None].expand(-1, c, -1))
    windows = windows.reshape(b, c, indices.shape[1], context**2)
    if channel_mode == "rgb":
        return windows.permute(0, 2, 1, 3).reshape(1, -1, c * context**2)
    return windows.permute(1, 0, 2, 3).reshape(c, -1, context**2)


class Moments:
    """Per-component sufficient statistics; all accumulators use float64."""

    def __init__(self, outputs, device="cpu"):
        self.count = 0
        self.total = torch.zeros(outputs, dtype=torch.float64, device=device)
        self.square = torch.zeros_like(self.total)
        self.sse = torch.zeros_like(self.total)

    def update(self, target, predicted=None):
        target = target.double()
        self.count += target.shape[0]
        self.total += target.sum(0)
        self.square += target.square().sum(0)
        if predicted is not None:
            self.sse += (target - predicted.double()).square().sum(0)

    def state(self):
        return {"count": self.count, "total": self.total.cpu(), "square": self.square.cpu()}

    def rows(self, channels, rank, train_mean):
        if not self.count:
            raise ValueError("No coefficient samples")
        mean = self.total / self.count
        second = self.square / self.count
        variance = (second - mean.square()).clamp_min(0)
        error = self.sse / self.count
        train_mean = train_mean.to(mean)
        mean_baseline = (second - 2 * train_mean * mean + train_mean.square()).clamp_min(0)
        rows = []
        for i in range(channels * rank):
            rows.append({"channel": i // rank, "component": i % rank + 1, "samples": self.count,
                         "target_mean": mean[i].item(), "target_variance": variance[i].item(),
                         "target_second_moment": second[i].item(), "mse": error[i].item(),
                         "r2": (1 - error[i] / variance[i]).item() if variance[i] > 1e-14 else None,
                         "train_mean_baseline_mse": mean_baseline[i].item(),
                         "gain_vs_train_mean": (mean_baseline[i] - error[i]).item(),
                         "gain_vs_zero": (second[i] - error[i]).item()})
        return rows


class StreamingRidge:
    def __init__(self, channels, rank, context, channel_mode, device):
        self.channels, self.rank, self.context, self.channel_mode = channels, rank, context, channel_mode
        groups = channels if channel_mode == "same_channel" else 1
        features = context**2 * (1 if groups == channels else channels) + 1
        outputs = rank if groups == channels else channels * rank
        self.xtx = torch.zeros(groups, features, features, dtype=torch.float64, device=device)
        self.xty = torch.zeros(groups, features, outputs, dtype=torch.float64, device=device)
        self.count = 0

    def update(self, lr, target, indices):
        x = sampled_windows(lr, self.context, indices, self.channel_mode).double()
        x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        y = sampled_values(target, indices).double()
        y = (y.reshape(-1, self.channels, self.rank).permute(1, 0, 2)
             if self.channel_mode == "same_channel" else y.unsqueeze(0))
        self.xtx += x.transpose(1, 2) @ x
        self.xty += x.transpose(1, 2) @ y
        self.count += x.shape[1]

    def solve(self, alpha):
        if alpha <= 0 or not self.count:
            raise ValueError("Ridge alpha and sample count must be positive")
        penalty = torch.eye(self.xtx.shape[-1], device=self.xtx.device, dtype=torch.float64)
        penalty[-1, -1] = 0  # Do not regularize the intercept.
        weights = torch.linalg.solve(self.xtx / self.count + alpha * penalty,
                                     self.xty / self.count)
        model = RidgePredictor(self.channels, self.rank, self.context, self.channel_mode)
        with torch.no_grad():
            model.conv.weight.copy_(weights[:, :-1].transpose(1, 2).reshape_as(model.conv.weight))
            model.conv.bias.copy_(weights[:, -1].reshape(-1))
        return model


class RidgePredictor(nn.Module):
    def __init__(self, channels, rank, context, channel_mode):
        super().__init__()
        self.channels, self.rank = channels, rank
        self.conv = nn.Conv2d(channels, channels * rank, context, padding=context // 2,
                              padding_mode="replicate", groups=channels if channel_mode == "same_channel" else 1)

    def forward(self, lr):
        return self.conv(lr)


class ContextCNN(nn.Module):
    """Nonlinear probe with exact theoretical LR receptive field `context`."""

    def __init__(self, channels, rank, context, width=16):
        super().__init__()
        if context < 1 or context % 2 != 1 or width < 1:
            raise ValueError("context must be positive odd; width must be positive")
        self.channels, self.rank = channels, rank
        layers, incoming = [], channels
        for _ in range(max(1, (context - 1) // 2)):
            kernel = 1 if context == 1 else 3
            layers.extend((nn.Conv2d(incoming, width, kernel, padding=kernel // 2,
                                     padding_mode="replicate"), nn.ReLU()))
            incoming = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, channels * rank, 1)

    def forward(self, lr):
        return self.head(self.features(lr))
