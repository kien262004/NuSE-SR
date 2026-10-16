"""Image evaluation with optional PCA truncation and prediction diagnostics."""
from pathlib import Path

import torch

from .losses import coefficient_image_mse
from .metrics import rgb_mse, rgb_psnr, rgb_ssim
from .utils import save_rgb
from .data import supervised_batch


@torch.no_grad()
def evaluate(model, loader, device, *, shave=0, image_dir=None, on_image=None):
    was_training = model.training
    model.eval()
    rows = []
    total_sse, total_pixels = 0.0, 0
    try:
        for batch in loader:
            lr, hr = supervised_batch(batch, device, channels=model.channels,
                                      scale=model.scale, operator=model.operator)
            if hr.shape[0] != 1:
                raise ValueError("Evaluation requires batch size one")
            if model.uses_null_space:
                predicted = model.predict_coefficients(lr)
                sr = model.operator.pseudoinverse(lr) + model.decoder(predicted)
            else:
                sr = model(lr)
            if not torch.isfinite(sr).all():
                raise FloatingPointError(f"Non-finite output for {batch['id'][0]}")
            full_mse = rgb_mse(sr, hr).item()
            row = {"name": batch["id"][0],
                   "rgb_psnr": rgb_psnr(sr, hr, shave).item(),
                   "rgb_ssim": rgb_ssim(sr, hr, shave).item(),
                   "rgb_mse": rgb_mse(sr, hr, shave).item(),
                   "full_image_mse": full_mse,
                   "sr_min": sr.min().item(),
                   "sr_max": sr.max().item(),
                   "sr_mean": sr.mean().item(),
                   "sr_below_zero_fraction": (sr < 0).float().mean().item(),
                   "sr_above_one_fraction": (sr > 1).float().mean().item()}
            if model.uses_null_space:
                oracle_coeffs = model.decoder.encode_hr(hr)
                oracle = model.operator.pseudoinverse(lr) + model.decoder(oracle_coeffs)
                truncation = rgb_mse(oracle, hr).item()
                prediction = coefficient_image_mse(predicted.double(), oracle_coeffs.double(),
                                                   channels=model.channels, scale=model.scale).item()
                row.update({"oracle_psnr": rgb_psnr(oracle, hr, shave).item(),
                            "truncation_mse": truncation,
                            "coefficient_image_mse": prediction,
                            "coefficient_mse": (predicted.double() - oracle_coeffs.double()).square().mean().item(),
                            "dc_max_abs": (model.operator(sr) - lr).abs().max().item(),
                            "decomposition_abs": abs(full_mse - truncation - prediction)})
            rows.append(row)
            total_sse += full_mse * hr.numel()
            total_pixels += hr.numel()
            if image_dir:
                save_rgb(Path(image_dir) / f"{len(rows):05d}_{Path(row['name']).stem}.png", sr[0])
            if on_image:
                on_image(row, sr[0], hr[0])
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("Evaluation loader is empty")
    summary = {key: sum(row[key] for row in rows) / len(rows)
               for key in rows[0] if key != "name"}
    for key in ("dc_max_abs", "decomposition_abs"):
        if key in summary:
            summary[key] = max(row[key] for row in rows)
    summary.update({"images": len(rows), "shave": shave,
                    "global_pixel_mse": total_sse / total_pixels,
                    "aggregation": ("image mean; DC/decomposition maximum; global_pixel_mse pixel weighted"
                                    if model.uses_null_space else
                                    "image mean; global_pixel_mse pixel weighted"),
                    "output_protocol": "raw RGB float, no clamp or quantization"})
    return summary, rows
