"""Baseline dựng lại từ thí nghiệm cũ: SPANV2, DISP, PLKSR-Tiny, và bản cắm NuSE k=8."""
import torch
from torch import nn
from torch.nn import functional as F

from ..core.constants import S
from ..operators import AvgPoolOp

class SPABV2(nn.Module):
    """SPANV2 block as in the previous experiments: y = (x + f3) * Conv1x1(f3)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.c3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gate = nn.Conv2d(out_channels, out_channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        f3 = self.c3(self.act(self.c2(self.act(self.c1(x)))))
        return (self.shortcut(x) + f3) * self.gate(f3)


class SPANV2(nn.Module):
    """SPANV2 (NTIRE 2026 ESR, XiaomiMM), same code as the previous SPANV2 runs."""

    def __init__(self, channels=3, rank=None, width=32, num_blocks=5):
        super().__init__()
        near = channels * S * S
        self.conv_near = nn.Conv2d(channels, near, 3, padding=1, groups=channels)
        with torch.no_grad():
            self.conv_near.weight.zero_()
            self.conv_near.weight[:, 0, 1, 1] = 1
            self.conv_near.bias.zero_()
        self.blocks = nn.Sequential(*(SPABV2(channels if i == 0 else width, width) for i in range(num_blocks)))
        fused = near + width
        self.fuse_depthwise = nn.Conv2d(fused, fused, 3, padding=1, groups=fused)
        self.fuse_pointwise = nn.Conv2d(fused, channels * S * S if rank is None else channels * rank, 1)
        self.upsample = nn.PixelShuffle(S) if rank is None else nn.Identity()

    def forward(self, lr):
        x = torch.cat((self.conv_near(lr), self.blocks(lr)), dim=1)
        return self.upsample(self.fuse_pointwise(self.fuse_depthwise(x)))


class DISPConv(nn.Module):
    def __init__(self, c_in, c_out, s=1, bias=True):
        super().__init__()
        self.eval_conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=s, bias=bias)

    def forward(self, x):
        return self.eval_conv(x)


class REECB(nn.Module):
    def __init__(self, in_channels, bias=False, extern_conv=0):
        super().__init__()
        self._extern_conv_block = nn.ModuleList(DISPConv(in_channels, in_channels, bias=bias)
                                                for _ in range(extern_conv))
        self.c1_r = DISPConv(in_channels, in_channels, bias=bias)
        self.c2_r = DISPConv(in_channels, in_channels, bias=bias)
        self.c3_r = DISPConv(in_channels, in_channels, bias=bias)
        self.act1 = nn.SiLU(inplace=True)

    def forward(self, x):
        out2 = self.act1(self.c2_r(self.act1(self.c1_r(x))))
        for extern in self._extern_conv_block:
            out2 = self.act1(extern(out2))
        return self.act1(self.c3_r(out2)) + x


class DISPNet(nn.Module):
    def __init__(self, num_in_ch=3, feature_channels=32, upscale=4, out_channels=None):
        super().__init__()
        out_channels = num_in_ch * upscale ** 2 if out_channels is None else out_channels
        self.conv_1 = DISPConv(num_in_ch, feature_channels, bias=True)
        self.blocks = nn.Sequential(*[REECB(feature_channels, bias=True, extern_conv=1) for _ in range(4)])
        self.conv_2 = nn.Conv2d(feature_channels, feature_channels, 1, 1, 0)
        conv = nn.Conv2d(feature_channels, out_channels, 3, padding=1)
        self.upsampler = (nn.Sequential(conv, nn.PixelShuffle(upscale))
                          if out_channels == num_in_ch * upscale ** 2 else conv)

    def forward(self, x):
        feat = self.conv_1(x)
        out = self.conv_2(self.blocks(feat)) + feat
        return self.upsampler(out)


class DISPSR(nn.Module):
    def __init__(self, channels=3, rank=None, width=32):
        super().__init__()
        self.body = DISPNet(channels, width, S, None if rank is None else channels * rank)

    def forward(self, lr):
        return self.body(lr)


class PLKConv2d(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        nn.init.trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x):
        x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
        return torch.cat([self.conv(x1), x2], dim=1)


class PLKBlock(nn.Module):
    def __init__(self, dim, kernel_size, split_ratio):
        super().__init__()
        self.channe_mixer = nn.Sequential(nn.Conv2d(dim, dim * 2, 3, 1, 1), nn.GELU(),
                                          nn.Conv2d(dim * 2, dim, 3, 1, 1))
        self.lk = PLKConv2d(int(dim * split_ratio), kernel_size)
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        return x + self.refine(self.lk(self.channe_mixer(x)))


class PLKSRNet(nn.Module):
    def __init__(self, dim=64, n_blocks=12, kernel_size=13, split_ratio=0.25, out_channels=48):
        super().__init__()
        self.out_channels = out_channels
        self.feats = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1),
                                   *[PLKBlock(dim, kernel_size, split_ratio) for _ in range(n_blocks)],
                                   nn.Conv2d(dim, out_channels, 3, 1, 1))
        nn.init.trunc_normal_(self.feats[0].weight, std=0.02)
        nn.init.trunc_normal_(self.feats[-1].weight, std=0.02)

    def forward(self, x):
        if self.out_channels == 3 * S * S:
            return F.pixel_shuffle(self.feats(x) + x.repeat_interleave(S * S, dim=1), S)
        return self.feats(x)


class PLKSRTiny(nn.Module):
    def __init__(self, channels=3, rank=None, width=64):
        super().__init__()
        self.body = PLKSRNet(dim=width, out_channels=channels * S * S if rank is None else channels * rank)

    def forward(self, lr):
        return self.body(lr)


BACKBONES = {"SPANV2": (SPANV2, 32), "DISP": (DISPSR, 32), "PLKSR-Tiny": (PLKSRTiny, 64)}


class ImageSR(nn.Module):
    """Plain baseline trained with image MSE. State keys `predictor.*` match the previous checkpoints."""
    kind = "image"

    def __init__(self, name):
        super().__init__()
        cls, width = BACKBONES[name]
        self.predictor = cls(3, None, width)

    def forward(self, lr, k=None):
        return self.predictor(lr)

    def loss(self, lr, hr, alpha=None):
        return F.mse_loss(self.predictor(lr), hr)


class LegacyNuSEPlugin(nn.Module):
    """Previous `backbone + NuSE (k=8)` runs: coefficient head + fixed rank-8 decoder + A+y (average pool).

    Only used to load and re-evaluate the previous checkpoints; buffer names match them.
    """
    kind = "legacy_plugin"

    def __init__(self, name, state):
        super().__init__()
        cls, width = BACKBONES[name]
        rank = state["decoder.components"].shape[-1]
        self.predictor = cls(3, rank, width)
        self.decoder = nn.Module()
        for key in ("Q", "mean", "components", "eigenvalues", "synthesis_basis", "block_mean"):
            self.decoder.register_buffer(key, state[f"decoder.{key}"].clone())
        self.rank = rank
        self.operator = AvgPoolOp()

    def forward(self, lr, k=None):
        coeffs = self.predictor(lr)
        b, _, h, w = coeffs.shape
        coeffs = coeffs.reshape(b, 3, self.rank, h, w)
        blocks = torch.einsum("bckhw,cpk->bcphw", coeffs, self.decoder.synthesis_basis.to(coeffs.dtype))
        blocks = blocks + self.decoder.block_mean[None, :, :, None, None].to(coeffs.dtype)
        return self.operator.pinv(lr) + F.pixel_shuffle(blocks.reshape(b, 3 * 16, h, w), S)
