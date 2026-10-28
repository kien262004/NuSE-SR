"""Đánh giá một model trên một tập ảnh: metric từng ảnh và trung bình."""
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..core.constants import S
from ..data.images import as_float, save_png
from .metrics import psnr, rgb_mse, ssim, to_y
from ..models.nuse_e import NuSEE
from ..operators import degrade, make_operator

@torch.no_grad()
def evaluate(model, images, method, device, *, k=None, lpips=None, fid=None, fid_real=False,
             save_dir=None, amp=False):
    """Per-image metrics on full images. SR is never clamped for PSNR/SSIM (raw float RGB)."""
    op = make_operator(method)
    rows = []
    model.eval()
    for hr_cpu, name in images:
        hr = as_float(hr_cpu, device)
        lr = degrade(hr, method)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            sr = model(lr, k) if isinstance(model, NuSEE) else model(lr)
        sr = sr.float()
        back = op.forward(sr)
        row = {"name": name, "rgb_psnr": psnr(sr, hr).item(), "rgb_ssim": ssim(sr, hr).item(),
               "y_psnr_shave4": psnr(to_y(sr), to_y(hr), S).item(), "y_ssim_shave4": ssim(to_y(sr), to_y(hr), S).item(),
               "rgb_psnr_uint8": psnr(sr.clamp(0, 1).mul(255).round().div(255), hr).item(),
               "lr_psnr": psnr(back, lr).item(), "lr_max_abs": (back - lr).abs().max().item(),
               "sr_below0": (sr < 0).float().mean().item(), "sr_above1": (sr > 1).float().mean().item()}
        if lpips is not None:
            row["lpips"] = lpips(sr, hr).item()
        if fid is not None:
            fid.update(sr, real=False)
            if fid_real:
                fid.update(hr, real=True)
        if isinstance(model, NuSEE):
            coeffs = model.coefficients(lr, k)
            cstar = model.decoder.targets(model.residual_blocks(hr))[:, :, :k]
            oracle = model.consistency(model.detail(cstar.flatten(1, 2), k), lr)
            row["oracle_psnr"] = psnr(oracle, hr).item()
            row["truncation_mse"] = rgb_mse(oracle, hr).item()
            row["coefficient_image_mse"] = ((coeffs.reshape_as(cstar) - cstar).double().square().sum() / hr.numel()).item()
        if save_dir is not None:
            save_png(Path(save_dir) / f"{name}.png", sr[0])
        rows.append(row)
    summary = {key: statistics.fmean(r[key] for r in rows) for key in rows[0] if key != "name"}
    summary["lr_max_abs"] = max(r["lr_max_abs"] for r in rows)
    if fid is not None:
        summary["fid_patch256"] = fid.compute_and_reset()
    summary["images"] = len(rows)
    return summary, rows


@torch.no_grad()
def evaluate_bicubic_upsampling(images, method, device, lpips=None, fid=None, fid_real=False):
    class Up(nn.Module):
        def forward(self, lr):
            return F.interpolate(lr, scale_factor=S, mode="bicubic", align_corners=False).clamp(0, 1)
    return evaluate(Up().to(device), images, method, device, lpips=lpips, fid=fid, fid_real=fid_real)
