"""Train the coefficient predictor with a fixed block-null-space PCA decoder."""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

import torch

from nullspaceSR.data import (build_dataset, make_loader, add_dataset_arguments,
                   dataset_manifest, set_dataset_epoch)
from nullspaceSR.engine import train_one_epoch
from nullspaceSR.evaluation import evaluate
from nullspaceSR.models import IMDN, NuSESR
from nullspaceSR.pca import BlockNullSpacePCA
from nullspaceSR.utils import (get_device, model_from_checkpoint, parse_args, restore_rng,
                    rng_state, save_checkpoint, seed_everything, sha256, write_csv, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-hr")
    parser.add_argument("--val-hr")
    parser.add_argument("--basis")
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    args = parse_args(parser)
    if not args.basis:
        parser.error("basis is required (CLI or YAML)")
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
    decoder = BlockNullSpacePCA.load(args.basis)
    if (decoder.scale, decoder.rank) != (args.scale, args.rank):
        raise ValueError("Basis scale/rank do not match training configuration")
    model = NuSESR(IMDN(decoder.channels, args.rank, args.width), decoder)
    basis_hash = sha256(args.basis)
    checkpoint = None
    start_epoch, best_psnr, history = 1, float("-inf"), []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
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
    architecture = {"name": "NuSESR", "scale": args.scale, "rank": args.rank,
                    "channels": decoder.channels, "width": args.width}
    print(f"Training {architecture} on {device}; {len(train_loader)} batches/epoch", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        set_dataset_epoch(train_data, epoch)
        train_loader.generator.manual_seed(args.seed + epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(model, train_loader, optimizer, device,
                                       grad_clip=args.grad_clip, epoch=epoch, log_every=args.log_every)
        record = {"epoch": epoch, "lr": current_lr, **train_metrics}
        improved = False
        if epoch % args.val_every == 0 or epoch == args.epochs:
            summary, rows = evaluate(model, val_loader, device, shave=args.shave)
            record["validation"] = summary
            write_json(output / "validation" / f"epoch_{epoch:04d}.json", summary)
            write_csv(output / "validation" / f"epoch_{epoch:04d}.csv", rows)
            improved = summary["rgb_psnr"] > best_psnr
            if improved:
                best_psnr = summary["rgb_psnr"]
            print(f"epoch {epoch}: val PSNR={summary['rgb_psnr']:.4f}, DC={summary['dc_max_abs']:.3e}", flush=True)
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
        print(f"epoch {epoch} complete; train loss={record['train_coefficient_image_mse']:.6f}", flush=True)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
