"""RGB metrics on float tensors; no implicit clamp or quantization."""
import torch
from torch import Tensor
from torch.nn import functional as F


def _crop_pair(predicted: Tensor, target: Tensor, shave: int = 0):
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("Metrics require matching BCHW tensors")
    if shave < 0 or min(predicted.shape[-2:]) <= 2 * shave:
        raise ValueError("Invalid border shave")
    if shave:
        predicted = predicted[..., shave:-shave, shave:-shave]
        target = target[..., shave:-shave, shave:-shave]
    return predicted, target


def rgb_mse(predicted: Tensor, target: Tensor, shave: int = 0) -> Tensor:
    predicted, target = _crop_pair(predicted, target, shave)
    return (predicted.double() - target.double()).square().flatten(1).mean(1)


def rgb_psnr(predicted: Tensor, target: Tensor, shave: int = 0,
             data_range: float = 1.0) -> Tensor:
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    # Identical images yield +inf, rather than an artificial capped PSNR.
    return 10 * torch.log10(data_range**2 / rgb_mse(predicted, target, shave))


def rgb_ssim(predicted: Tensor, target: Tensor, shave: int = 0,
             data_range: float = 1.0) -> Tensor:
    """RGB SSIM: Gaussian sigma=1.5, valid window, population covariance.

    Window is 11x11, reduced to the largest fitting odd size for small images.
    Returns one value per image, averaging channels and valid spatial positions.
    """
    predicted, target = _crop_pair(predicted, target, shave)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    predicted, target = predicted.float(), target.float()
    size = min(11, *predicted.shape[-2:])
    size = size if size % 2 else size - 1
    axis = torch.arange(size, device=predicted.device, dtype=predicted.dtype) - size // 2
    gaussian = torch.exp(-axis.square() / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :]).expand(predicted.shape[1], 1, size, size)
    def smooth(x):
        return F.conv2d(x, window, groups=predicted.shape[1])
    mx, my = smooth(predicted), smooth(target)
    vx = (smooth(predicted.square()) - mx.square()).clamp_min(0)
    vy = (smooth(target.square()) - my.square()).clamp_min(0)
    covariance = smooth(predicted * target) - mx * my
    c1, c2 = (0.01 * data_range)**2, (0.03 * data_range)**2
    values = ((2 * mx * my + c1) * (2 * covariance + c2)
              / ((mx.square() + my.square() + c1) * (vx + vy + c2)))
    return values.flatten(1).mean(1)
