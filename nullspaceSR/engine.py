"""Training epoch over the common batch contract, independent of dataset."""
import torch

from .data import supervised_batch
from .losses import coefficient_image_mse


def train_one_epoch(model, loader, optimizer, device, *, grad_clip=1.0,
                    epoch=1, log_every=100, on_step=None):
    model.train()
    weighted_loss, pixels = 0.0, 0
    for step, batch in enumerate(loader, 1):
        hr_channels, scale = model.decoder.channels, model.decoder.scale
        lr, hr = supervised_batch(batch, device, channels=hr_channels,
                                  scale=scale, operator=model.operator)
        with torch.no_grad():
            target = model.decoder.encode_hr(hr)
        predicted = model.predict_coefficients(lr)
        loss = coefficient_image_mse(predicted, target, channels=hr_channels, scale=scale)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: epoch={epoch}, step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        weighted_loss += loss.item() * hr.numel()
        pixels += hr.numel()
        if log_every and (step % log_every == 0 or step == len(loader)):
            if step % log_every == 0:
                print(f"epoch {epoch} batch {step} loss={loss.item():.6f}", flush=True)
            if on_step:
                on_step(step, len(loader), loss.item())
    if not pixels:
        raise ValueError("Training loader is empty")
    return {"train_coefficient_image_mse": weighted_loss / pixels}
