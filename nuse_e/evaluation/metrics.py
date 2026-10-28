"""PSNR, SSIM (RGB hoặc kênh Y, có shave)."""
import torch
from torch.nn import functional as F

def _crop(a, b, shave):
    if shave:
        a, b = a[..., shave:-shave, shave:-shave], b[..., shave:-shave, shave:-shave]
    return a, b


def rgb_mse(sr, hr, shave=0):
    sr, hr = _crop(sr, hr, shave)
    return (sr.double() - hr.double()).square().flatten(1).mean(1)


def psnr(sr, hr, shave=0):
    return 10 * torch.log10(1.0 / rgb_mse(sr, hr, shave))


def ssim(sr, hr, shave=0):
    """Gaussian 11x11, sigma 1.5, valid window, population covariance (same as the repo)."""
    sr, hr = _crop(sr, hr, shave)
    sr, hr = sr.float(), hr.float()
    size = min(11, *sr.shape[-2:])
    size = size if size % 2 else size - 1
    axis = torch.arange(size, device=sr.device, dtype=sr.dtype) - size // 2
    g = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(sr.shape[1], 1, size, size)

    def smooth(x):
        return F.conv2d(x, window, groups=sr.shape[1])
    mx, my = smooth(sr), smooth(hr)
    vx = (smooth(sr.square()) - mx.square()).clamp_min(0)
    vy = (smooth(hr.square()) - my.square()).clamp_min(0)
    cov = smooth(sr * hr) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * cov + c2)) / ((mx.square() + my.square() + c1) * (vx + vy + c2))).flatten(1).mean(1)


def to_y(x):
    """uint8-rounded RGB -> Y (BT.601), range [0,1]."""
    x = x.clamp(0, 1).mul(255).round().div(255)
    return ((65.481 * x[:, 0] + 128.553 * x[:, 1] + 24.966 * x[:, 2] + 16) / 255)[:, None]
