"""Evaluate a self-contained NuSESR checkpoint on an HR image folder."""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

import torch

from nullspaceSR.data import build_dataset, make_loader, add_dataset_arguments, dataset_manifest
from nullspaceSR.evaluation import evaluate
from nullspaceSR.utils import get_device, model_from_checkpoint, parse_args, sha256, write_csv, write_json
from nullspaceSR.tracking import add_wandb_arguments, init_wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--hr-dir")
    add_dataset_arguments(parser, "test")
    parser.add_argument("--output", default="results/nusesr")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--shave", type=int, default=0)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=False)
    add_wandb_arguments(parser)
    args = parse_args(parser, stage="evaluation")
    if not args.checkpoint:
        parser.error("checkpoint is required (CLI or YAML)")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device = get_device(args.device)
    model = model_from_checkpoint(checkpoint).to(device)
    dataset = build_dataset(args.test_dataset, args.test_dataset_options,
                            folder_defaults={"hr_dir": args.hr_dir, "scale": model.decoder.scale})
    loader = make_loader(dataset, batch_size=1, workers=args.workers, pin_memory=device.type == "cuda")
    output = Path(args.output)
    summary, rows = evaluate(model, loader, device, shave=args.shave,
                             image_dir=output / "images" if args.save_images else None)
    report = {"summary": summary, "config": vars(args), "architecture": checkpoint["architecture"],
              "checkpoint_sha256": sha256(args.checkpoint),
              "manifest": dataset_manifest(dataset), "torch_version": str(torch.__version__)}
    write_json(output / "metrics.json", report)
    write_csv(output / "per_image.csv", rows)
    print(f"{summary['images']} images: RGB PSNR={summary['rgb_psnr']:.4f}, "
          f"SSIM={summary['rgb_ssim']:.6f}, "
          f"SR range=[{summary['sr_min']:.4f}, {summary['sr_max']:.4f}], "
          f"below0={summary['sr_below_zero_fraction']:.1%}, "
          f"above1={summary['sr_above_one_fraction']:.1%}, "
          f"DC max={summary['dc_max_abs']:.3e}")
    print(f"Results: {output}")
    wandb, wandb_run = init_wandb(args, job_type="test")
    if wandb_run is not None:
        wandb_run.log({f"test/{key}": value for key, value in summary.items()
                       if isinstance(value, (int, float))})
        table = wandb.Table(columns=["image", "rgb_psnr", "rgb_ssim", "rgb_mse"])
        for row in rows:
            table.add_data(row["name"], row["rgb_psnr"], row["rgb_ssim"], row["rgb_mse"])
        wandb_run.log({"test/per_image_metrics": table})
        wandb_run.finish()


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
