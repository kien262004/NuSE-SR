"""Vòng train một run: AdamW, cosine theo epoch, validate, best.pt / last.pt / DONE."""
import json
import os
import statistics
import time
from pathlib import Path

import torch

from ..core.constants import S
from ..core.io import seed_everything, write_json
from ..data.cache import FlatImages
from ..data.images import ValImages
from ..data.loaders import RamCropLoader, train_loader
from ..evaluation.evaluate import evaluate
from ..models.nuse_e import NuSEE
from ..models.registry import budgets_of, build_model
from ..operators import degrade

def quick_images(val, count, crop):
    """First `count` validation images, centre-cropped to crop x crop: fast per-epoch progress check."""
    out = []
    for i in range(min(count, len(val))):
        hr, name = val[i]
        c = min(crop, hr.shape[-2], hr.shape[-1]) // S * S
        top, left = (hr.shape[-2] - c) // 2 // S * S, (hr.shape[-1] - c) // 2 // S * S
        out.append((hr[:, top:top + c, left:left + c].contiguous(), name))
    return out


def progress_text(metrics, model):
    if isinstance(model, NuSEE):
        return " | ".join(f"K{k}: {metrics[f'rgb_psnr_K{k}']:.3f} dB / {metrics[f'rgb_ssim_K{k}']:.4f}" for k in model.budgets)
    return f"{metrics['rgb_psnr']:.3f} dB / {metrics['rgb_ssim']:.4f}"


@torch.no_grad()
def validate(model, images, method, device):
    out = {}
    for k in budgets_of(model):
        summary, _ = evaluate(model, images, method, device, k=k)
        out["rgb_psnr" if k is None else f"rgb_psnr_K{k}"] = summary["rgb_psnr"]
        out["rgb_ssim" if k is None else f"rgb_ssim_K{k}"] = summary["rgb_ssim"]
    if isinstance(model, NuSEE):
        out["rgb_psnr"] = statistics.fmean(out[f"rgb_psnr_K{k}"] for k in model.budgets)
        out["rgb_ssim"] = statistics.fmean(out[f"rgb_ssim_K{k}"] for k in model.budgets)
    return out


def train(cfg, log=print):
    """One run. cfg keys: model, method, run_dir, train_files, val_files, stats, arch, epochs, batch,
    repeat, lr, min_lr, grad_clip, val_every, checkpoint_every, workers, seed, amp, device, max_iters."""
    device = torch.device(cfg["device"])
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    stats = torch.load(cfg["stats"], map_location="cpu", weights_only=False) if cfg.get("stats") else None
    arch = json.loads(Path(cfg["arch"]).read_text()) if cfg.get("arch") else None
    model = build_model(cfg["model"], cfg["method"], stats, arch).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["min_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    if cfg.get("train_cache"):                           # flat uint8 cache in RAM: no image decoding per crop
        loader = RamCropLoader(FlatImages(cfg["train_cache"]), cfg["batch"], cfg["seed"], repeat=cfg["repeat"])
    else:
        data, dl = train_loader([Path(p) for p in cfg["train_files"]], batch=cfg["batch"], workers=cfg["workers"],
                                seed=cfg["seed"], repeat=cfg["repeat"])

        class _Wrapped:
            def set_epoch(self, epoch):
                data.set_epoch(epoch)
                dl.generator.manual_seed(cfg["seed"] + epoch)

            def __iter__(self):
                for hr, idx in dl:
                    yield hr.mul(255).round().byte(), idx
        loader = _Wrapped()
    val = ValImages([Path(p) for p in cfg["val_files"]])
    val_images = [val[i] for i in range(len(val))]
    quick = quick_images(val, cfg.get("quick_val_images", 10), cfg.get("quick_val_crop", 480))
    start, best, history = 1, -float("inf"), []
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        log(f"resume from epoch {ck['epoch']}")
    meta = {"model": cfg["model"], "method": cfg["method"], "protocol": {k: cfg[k] for k in (
        "epochs", "batch", "repeat", "lr", "min_lr", "grad_clip", "seed", "amp")}}
    t_run = time.time()
    for epoch in range(start, cfg["epochs"] + 1):
        model.train()
        loader.set_epoch(epoch)
        total, count, t0, t_data = 0.0, 0, time.time(), 0.0
        t_step = time.time()
        for step, (hr, _) in enumerate(loader, 1):
            t_data += time.time() - t_step
            hr = hr.to(device, non_blocking=True).float().div_(255)
            with torch.no_grad():
                lr = degrade(hr, cfg["method"])
            with torch.autocast(device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type == "cuda"):
                loss = model.loss(lr, hr)
            loss = loss.float()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            total += loss.item() * hr.shape[0]
            count += hr.shape[0]
            t_step = time.time()
            if cfg.get("max_iters") and step >= cfg["max_iters"]:
                break
        record = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": total / count,
                  "epoch_seconds": time.time() - t0, "data_seconds": t_data}
        sched.step()
        # Progress check every epoch on a few centre crops: no gradient, no RNG use, so training is unchanged.
        record["quick"] = validate(model, quick, cfg["method"], device)
        model.train()
        improved = False
        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
            record["validation"] = validate(model, val_images, cfg["method"], device)
            improved = record["validation"]["rgb_psnr"] > best
            best = max(best, record["validation"]["rgb_psnr"])
        history.append(record)
        state = {"nuse_e": meta, "epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "best": best,
                 "history": history, "config": cfg, "stats": stats, "arch": arch}
        tmp = run_dir / "last.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, last)
        if improved:
            torch.save(state, run_dir / "best.pt")
        if epoch % cfg["checkpoint_every"] == 0:
            torch.save(state, run_dir / f"epoch_{epoch:04d}.pt")
        write_json(run_dir / "history.json", history)
        msg = (f"epoch {epoch}/{cfg['epochs']} loss={record['train_loss']:.6f} "
               f"({record['epoch_seconds']:.0f}s, chờ dữ liệu {t_data:.0f}s) "
               f"| nhanh ({len(quick)} crop): {progress_text(record['quick'], model)}")
        if "validation" in record:
            msg += f" || DIV2K valid đủ {len(val_images)} ảnh: {progress_text(record['validation'], model)}"
        log(msg)
    (run_dir / "DONE").write_text(f"{time.time() - t_run:.0f}s\n")
