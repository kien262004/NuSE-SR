"""Evaluate a self-contained NuSESR checkpoint on an HR image folder."""
import argparse
from pathlib import Path

import torch

from .data import build_dataset, make_loader, add_dataset_arguments, dataset_manifest
from .evaluation import evaluate
from .utils import get_device, model_from_checkpoint, parse_args, sha256, write_csv, write_json


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
    args = parse_args(parser)
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
          f"SSIM={summary['rgb_ssim']:.6f}, DC max={summary['dc_max_abs']:.3e}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
