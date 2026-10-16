"""Low-resolution image generation functions for HR-backed datasets."""
import importlib

import torch
from torch.nn import functional as F


def average_pool(hr, scale):
    """Apply NuSESR's non-overlapping block-average measurement operator."""
    return F.avg_pool2d(hr, kernel_size=scale, stride=scale)


def bicubic(hr, scale, *, size=None, antialias=True):
    """Downsample by bicubic interpolation with antialiasing."""
    return F.interpolate(hr.unsqueeze(0), size=size,
                         scale_factor=None if size is not None else 1 / scale,
                         mode="bicubic", align_corners=False, antialias=antialias).squeeze(0)


def bilinear(hr, scale, *, size=None, antialias=True):
    return F.interpolate(hr.unsqueeze(0), size=size,
                         scale_factor=None if size is not None else 1 / scale,
                         mode="bilinear", align_corners=False, antialias=antialias).squeeze(0)


def resolve_downsampler(spec):
    """Resolve a built-in downsampler or custom ``module:function`` callable."""
    if callable(spec):
        return spec, f"{spec.__module__}:{spec.__qualname__}"
    if not isinstance(spec, str) or not spec:
        raise ValueError("downsampler must be 'average_pool', 'bicubic', 'bilinear', or 'module:function'")
    if spec == "average_pool":
        return average_pool, spec
    if spec == "bicubic":
        return bicubic, spec
    if spec == "bilinear":
        return bilinear, spec
    if ":" not in spec:
        raise ValueError("Custom downsampler must use the 'module:function' format")
    module_name, function_name = spec.split(":", 1)
    if not module_name or not function_name:
        raise ValueError("Custom downsampler must use the 'module:function' format")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Configured downsampler is not callable: {spec}")
    return function, spec


def make_lr(hr, *, scale, downsampler, size=None, antialias=True):
    """Run a configured HR-to-LR function and validate its tensor contract."""
    if downsampler in (bicubic, bilinear):
        lr = downsampler(hr, scale, size=size, antialias=antialias)
    else:
        lr = downsampler(hr, scale)
    expected = (hr.shape[-2] // scale, hr.shape[-1] // scale)
    if not isinstance(lr, torch.Tensor) or lr.ndim != 3:
        raise TypeError("downsampler must return a CHW torch.Tensor")
    if lr.shape[0] != hr.shape[0] or tuple(lr.shape[-2:]) != expected:
        raise ValueError(f"downsampler returned shape {tuple(lr.shape)}; expected Cx{expected[0]}x{expected[1]}")
    if not lr.is_floating_point() or lr.device.type != "cpu":
        raise ValueError("downsampler output must be a floating CPU tensor")
    if not torch.isfinite(lr).all():
        raise ValueError("downsampler output contains non-finite values")
    return lr
