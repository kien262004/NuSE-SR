"""NuSE-E: trunk theo stage, head hệ số mỗi nhóm, decoder PCA cố định, nhất quán dữ liệu."""
import torch
from torch import nn
from torch.nn import functional as F

from ..core.constants import S
from .blocks import ResBlock
from ..nullspace.architecture import loss_alpha
from ..nullspace.basis import to_blocks
from ..operators import make_operator

class GroupHead(nn.Module):
    """c_G = sqrt(lambda_G) * (Conv1x1(f_g) [+ L_g(y)]); readout zero-initialised."""

    def __init__(self, width, channels, size, std, linear):
        super().__init__()
        self.channels, self.size = channels, size
        self.readout = nn.Conv2d(width, channels * size, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.linear = (nn.Conv2d(channels, channels * size, 3, padding=1, groups=channels,
                                 padding_mode="replicate") if linear else None)
        self.register_buffer("std", std.reshape(1, channels * size, 1, 1).float())

    @torch.no_grad()
    def set_linear(self, weights):
        """weights (C, size, 10): 9 row-major taps + bias, in coefficient units."""
        std = self.std.reshape(self.channels, self.size)
        w = weights.to(std) / std.unsqueeze(-1)
        self.linear.weight.copy_(w[..., :9].reshape(self.channels * self.size, 1, 3, 3))
        self.linear.bias.copy_(w[..., 9].reshape(-1))

    def forward(self, f, y):
        out = self.readout(f)
        if self.linear is not None:
            out = out + self.linear(y)
        return out * self.std


class FixedDecoder(nn.Module):
    """z = mu + U_K c per channel: one grouped 1x1 conv + PixelShuffle. Buffers only."""

    def __init__(self, mean_block, basis):
        super().__init__()
        self.channels = basis.shape[0]
        self.register_buffer("mean_block", mean_block.float())          # (C,16)
        self.register_buffer("basis", basis.float())                    # (C,16,d)

    def forward(self, coeffs, k):
        c = self.channels
        weight = self.basis[..., :k].reshape(c * S * S, k, 1, 1).to(coeffs.dtype)
        return F.pixel_shuffle(F.conv2d(coeffs, weight, self.mean_block.reshape(-1).to(coeffs.dtype), groups=c), S)

    def targets(self, residual_blocks):
        """c* = U^T (x_b - mu): (B,C,16,h,w) -> (B,C,d,h,w)."""
        return torch.einsum("bcphw,cpk->bckhw", residual_blocks - self.mean_block[None, :, :, None, None],
                            self.basis)


class NuSEE(nn.Module):
    kind = "nuse_e"

    def __init__(self, stats, arch, channels=3):
        super().__init__()
        width, kind = arch.get("width", 32), arch.get("block", "plain")
        self.width, self.kind, self.alpha = width, kind, arch.get("alpha", "uniform")
        self.method = stats["method"]
        self.groups, self.blocks_per_stage = tuple(arch["groups"]), tuple(arch["blocks"])
        self.budgets = list(arch["budgets"])
        self.channels = channels
        self.operator = make_operator(self.method)
        mean, vec, lam = stats["mean"], stats["components"], stats["eigenvalues"]
        if self.method == "average_pool":
            q = stats["Q"]
            basis = torch.einsum("pd,cdk->cpk", q, vec)                   # U = Q V
            mean_block = torch.einsum("pd,cd->cp", q, mean)              # mu = Q m
        else:
            basis, mean_block = vec, mean
        self.decoder = FixedDecoder(mean_block, basis)
        self.stem = nn.Conv2d(channels, width, 3, padding=1)
        self.stages = nn.ModuleList(nn.Sequential(*[ResBlock(width, kind) for _ in range(n)]) for n in arch["blocks"])
        std = lam.clamp_min(1e-12).sqrt()
        self.heads, start = nn.ModuleList(), 0
        for size, lin in zip(arch["groups"], arch["linear"]):
            head = GroupHead(width, channels, size, std[:, start:start + size], lin)
            if lin:
                head.set_linear(stats["wiener"][:, :, start:start + size].permute(0, 2, 1))
            self.heads.append(head)
            start += size

    def coefficients(self, y, k):
        g = self.budgets.index(k)
        f, parts = self.stem(y), []
        for j in range(g + 1):
            f = self.stages[j](f)
            parts.append(self.heads[j](f, y).reshape(y.shape[0], self.channels, self.groups[j], *y.shape[-2:]))
        return torch.cat(parts, 2).flatten(1, 2)

    def detail(self, coeffs, k):
        return self.decoder(coeffs, k)

    def consistency(self, z, y):
        if self.method == "average_pool":
            return z + self.operator.pinv(y)                             # Az = 0 (Thm. 2)
        return z + self.operator.pinv(y - self.operator.forward(z))     # Thm. 6

    def forward(self, y, k=None):
        k = self.budgets[-1] if k is None else k
        return self.consistency(self.detail(self.coefficients(y, k), k), y)

    def fuse(self):
        """Inference form: fold RepConv branches. Idempotent; a no-op for plain / lk7 blocks."""
        for stage in self.stages:
            for block in stage:
                block.fuse()
        return self

    def params_for(self, k):
        """Inference parameters (RepConv counted fused) of the model that serves budget k."""
        g = self.budgets.index(k)
        trunk = sum(ResBlock.inference_params(self.width, self.kind) for s in self.stages[:g + 1] for _ in s)
        mods = [self.stem, *self.heads[:g + 1]]
        return trunk + sum(p.numel() for m in mods for p in m.parameters())

    def residual_blocks(self, hr):
        if self.method == "average_pool":
            return to_blocks(hr)                     # U^T removes the block mean (1^T U = 0)
        return to_blocks(hr - self.operator.pinv(self.operator.forward(hr)))

    def loss(self, lr, hr, alpha=None):
        """Avg pool: weighted coefficient MSE (Thm. 4). Bicubic: sum of image MSE over budgets (Thm. 7)."""
        kmax = self.budgets[-1]
        alpha = loss_alpha(self.alpha if alpha is None else alpha, len(self.budgets))
        chat = self.coefficients(lr, kmax).float()
        with torch.autocast(lr.device.type, enabled=False):
            return self._loss(chat, lr.float(), hr.float(), kmax, alpha)

    def _loss(self, chat, lr, hr, kmax, alpha):
        if self.method == "average_pool":
            with torch.no_grad():
                cstar = self.decoder.targets(self.residual_blocks(hr))
            w = torch.zeros(kmax, device=lr.device)
            for k, a in zip(self.budgets, alpha):
                w[:k] += a
            err = (chat.reshape_as(cstar) - cstar).square()
            return (w[None, None, :, None, None] * err).sum() / hr.numel()
        b = lr.shape[0]
        chat = chat.reshape(b, self.channels, kmax, *lr.shape[-2:])
        total = 0.0
        for k, a in zip(self.budgets, alpha):
            sr = self.consistency(self.detail(chat[:, :, :k].flatten(1, 2), k), lr)
            total = total + a * F.mse_loss(sr, hr)
        return total
