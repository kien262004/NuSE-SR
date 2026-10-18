"""Train IMDN in image space or with the optional fixed PCA null-space plugin."""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

import torch

from nullspaceSR.data import (build_dataset, make_loader, add_dataset_arguments,
                   dataset_manifest, set_dataset_epoch)
from nullspaceSR.engine import train_one_epoch
from nullspaceSR.evaluation import evaluate
from nullspaceSR.metrics import LPIPSMetric
from nullspaceSR.models import IMDN, NuSESR
from nullspaceSR.pca import BlockNullSpacePCA
from nullspaceSR.utils import (get_device, model_from_checkpoint, parse_args, restore_rng,
                    rng_state, save_checkpoint, seed_everything, sha256, write_csv, write_json)
from nullspaceSR.tracking import add_wandb_arguments, init_wandb, wandb_rgb_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-hr")
    parser.add_argument("--val-hr")
    parser.add_argument("--basis")
    parser.add_argument("--null-space", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable PCA plugin (default: enabled when basis is supplied)")
    parser.add_argument("--channels", type=int, default=3)
    add_dataset_arguments(parser, "train")
    add_dataset_arguments(parser, "val")
    parser.add_argument("--output", default="runs/nusesr_x4_k8")
    parser.add_argument("--resume", help="Resume an epoch-boundary last.pt checkpoint")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--grad-clip", type=float, default=1)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--shave", type=int, default=0)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    add_wandb_arguments(parser)
    args = parse_args(parser, stage="train")
    if not isinstance(args.lpips, bool):
        parser.error("lpips must be true or false")
    if args.null_space is not None and not isinstance(args.null_space, bool):
        parser.error("null_space must be true or false")
    args.null_space = bool(args.basis) if args.null_space is None else args.null_space
    if args.null_space and not args.basis:
        parser.error("basis is required when null_space is enabled")
    if min(args.epochs, args.val_every, args.log_every, args.checkpoint_every) < 1:
        parser.error("epochs, val_every, log_every and checkpoint_every must be positive")
    if args.lr <= 0 or not 0 <= args.min_lr <= args.lr or args.weight_decay < 0 or args.grad_clip < 0:
        parser.error("Invalid optimizer parameters")
    output = Path(args.output)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError(f"{output}/last.pt exists; use --resume or a new output folder")
    seed_everything(args.seed)
    device = get_device(args.device)
    train_data = build_dataset(args.train_dataset, args.train_dataset_options,
                              folder_defaults={"hr_dir": args.train_hr, "scale": args.scale,
                                               "training": True, "crop_size": args.crop_size,
                                               "augment": args.augment, "repeat": args.repeat, "seed": args.seed})
    val_data = build_dataset(args.val_dataset, args.val_dataset_options,
                            folder_defaults={"hr_dir": args.val_hr, "scale": args.scale})
    manifest = {"train": dataset_manifest(train_data), "validation": dataset_manifest(val_data)}
    if set(manifest["train"].get("sources", [])) & set(manifest["validation"].get("sources", [])):
        raise ValueError("Train and validation image paths overlap")
    if args.null_space:
        decoder = BlockNullSpacePCA.load(args.basis)
        if (decoder.scale, decoder.rank) != (args.scale, args.rank):
            raise ValueError("Basis scale/rank do not match training configuration")
        model = NuSESR(IMDN(decoder.channels, args.rank, args.width), decoder)
        basis_hash = sha256(args.basis)
    else:
        model = NuSESR(IMDN(args.channels, width=args.width, scale=args.scale))
        basis_hash = None
    checkpoint = None
    start_epoch, best_psnr, history = 1, float("-inf"), []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["architecture"].get("null_space", True) != args.null_space:
            raise ValueError("Resume null_space mode differs from the checkpoint")
        if checkpoint["architecture"]["channels"] != model.channels:
            raise ValueError("Resume channels differ from the checkpoint")
        if checkpoint["basis_sha256"] != basis_hash:
            raise ValueError("Resume basis differs from the checkpoint basis")
        for key in ("scale", "rank", "width", "epochs", "seed", "batch_size", "crop_size",
                    "repeat", "augment", "lr", "min_lr", "weight_decay", "grad_clip", "shave", "val_every",
                    "train_dataset", "train_dataset_options", "val_dataset", "val_dataset_options"):
            if checkpoint["config"][key] != getattr(args, key):
                raise ValueError(f"Resume config changed: {key}; retain the original training schedule")
        if checkpoint["manifest"] != manifest:
            raise ValueError("Resume dataset manifest differs from the checkpoint")
        model = model_from_checkpoint(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_psnr, history = checkpoint["best_psnr"], checkpoint["history"]
    if start_epoch > args.epochs:
        print("Checkpoint already completed the configured training schedule.")
        return
    model.to(device)
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
    train_loader = make_loader(train_data, batch_size=args.batch_size, workers=args.workers,
                               shuffle=True, seed=args.seed, pin_memory=device.type == "cuda")
    val_loader = make_loader(val_data, batch_size=1, workers=args.workers,
                             pin_memory=device.type == "cuda")
    write_json(output / "config.json", vars(args))
    write_json(output / "manifest.json", manifest)
    architecture = {"name": "NuSESR" if args.null_space else "IMDN",
                    "null_space": args.null_space, "scale": args.scale,
                    "rank": args.rank if args.null_space else None,
                    "channels": model.channels, "width": args.width}
    print(f"Training {architecture} on {device}; {len(train_loader)} batches/epoch", flush=True)
    wandb, wandb_run = init_wandb(args, job_type="train")
    lpips_metric = LPIPSMetric().to(device) if args.lpips else None
    for epoch in range(start_epoch, args.epochs + 1):
        set_dataset_epoch(train_data, epoch)
        train_loader.generator.manual_seed(args.seed + epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        def log_batch(step, total_steps, loss, lr, hr):
            global_step = (epoch - 1) * len(train_loader) + step
            if wandb_run is not None and (step % args.log_every == 0 or step == total_steps):
                wandb_run.log({"global_step": global_step,
                               "epoch": epoch, "train/batch_loss": loss,
                               "train/batch": step, "train/batches_per_epoch": total_steps})
            if wandb_run is not None:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    preview_lr, preview_hr = lr[:1], hr[:1]
                    preview_sr = model(preview_lr)
                    preview_base = torch.nn.functional.interpolate(
                        preview_lr, scale_factor=model.scale, mode="nearest")
                model.train(was_training)
                wandb_run.log({
                    "global_step": global_step,
                    "train/lr_base": wandb_rgb_image(
                        wandb, preview_base[0], caption=f"Train step {global_step}: repeated LR"),
                    "train/sr": wandb_rgb_image(
                        wandb, preview_sr[0], caption=f"Train step {global_step}: SR"),
                    "train/hr": wandb_rgb_image(
                        wandb, preview_hr[0], caption=f"Train step {global_step}: HR target"),
                    "train/hr_min": preview_hr.min().item(),
                    "train/hr_max": preview_hr.max().item(),
                    "train/hr_mean": preview_hr.mean().item(),
                    "train/sr_min": preview_sr.min().item(),
                    "train/sr_max": preview_sr.max().item(),
                    "train/sr_mean": preview_sr.mean().item(),
                })

        train_metrics = train_one_epoch(model, train_loader, optimizer, device,
                                       grad_clip=args.grad_clip, epoch=epoch,
                                       log_every=args.log_every, on_step=log_batch)
        record = {"epoch": epoch, "lr": current_lr, **train_metrics}
        improved = False
        wandb_metrics = {"global_step": epoch * len(train_loader), "epoch": epoch,
                         "train/epoch_loss": train_metrics[model.loss_name],
                         "train/lr": current_lr}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            validation_images = []
            def collect_validation_image(row, sr, hr):
                if wandb_run is not None and len(validation_images) < 4:
                    validation_images.append((row["name"], sr.detach().cpu(), hr.detach().cpu()))

            summary, rows = evaluate(model, val_loader, device, shave=args.shave,
                                     lpips_metric=lpips_metric,
                                     on_image=collect_validation_image)
            record["validation"] = summary
            write_json(output / "validation" / f"epoch_{epoch:04d}.json", summary)
            write_csv(output / "validation" / f"epoch_{epoch:04d}.csv", rows)
            improved = summary["rgb_psnr"] > best_psnr
            if improved:
                best_psnr = summary["rgb_psnr"]
            print(f"epoch {epoch}: val PSNR={summary['rgb_psnr']:.4f}, "
                  f"SSIM={summary['rgb_ssim']:.6f}, "
                  f"SR range=[{summary['sr_min']:.4f}, {summary['sr_max']:.4f}], "
                  f"below0={summary['sr_below_zero_fraction']:.1%}, "
                  f"above1={summary['sr_above_one_fraction']:.1%}", flush=True)
            if "lpips" in summary:
                print(f"LPIPS={summary['lpips']:.6f}", flush=True)
            if model.uses_null_space:
                print(f"DC={summary['dc_max_abs']:.3e}", flush=True)
            wandb_metrics.update({f"validation/{key}": summary[key]
                                  for key in ("rgb_psnr", "rgb_ssim", "lpips", "rgb_mse", "oracle_psnr",
                                              "sr_min", "sr_max", "sr_mean",
                                              "sr_below_zero_fraction", "sr_above_one_fraction",
                                              "pca_truncation_mse", "measurement_mismatch_mse",
                                              "dc_max_abs", "decomposition_abs") if key in summary})
            for index, (name, sr, hr) in enumerate(validation_images, 1):
                wandb_metrics[f"validation/image_{index:02d}_sr"] = wandb_rgb_image(
                    wandb, sr, caption=f"Epoch {epoch}: {name} SR")
                wandb_metrics[f"validation/image_{index:02d}_hr"] = wandb_rgb_image(
                    wandb, hr, caption=f"Epoch {epoch}: {name} HR")
        if wandb_run is not None:
            wandb_run.log(wandb_metrics)
        history.append(record)
        scheduler.step()
        state = {"format_version": 2, "epoch": epoch, "architecture": architecture,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": rng_state(),
                 "best_psnr": best_psnr, "history": history, "config": vars(args),
                 "basis_sha256": basis_hash, "manifest": manifest,
                 "torch_version": str(torch.__version__)}
        save_checkpoint(output / "last.pt", state)
        if epoch % args.checkpoint_every == 0:
            save_checkpoint(output / f"epoch_{epoch:04d}.pt", state)
        if improved:
            save_checkpoint(output / "best.pt", state)
        write_json(output / "history.json", history)
        print(f"epoch {epoch} complete; train loss={record[model.loss_name]:.6f}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
