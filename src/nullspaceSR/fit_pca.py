"""CLI: fit a frozen decoder from train HR crops only."""
import argparse
from pathlib import Path

import torch

from .data import (build_dataset, make_loader, add_dataset_arguments,
                   dataset_manifest, supervised_batch)
from .pca import BlockNullSpacePCA
from .utils import parse_args, seed_everything, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-hr")
    add_dataset_arguments(parser, "train")
    parser.add_argument("--output", default="basis/avgpool_x4_rank8.pt")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    args = parse_args(parser)
    seed_everything(args.seed)
    dataset = build_dataset(args.train_dataset, args.train_dataset_options,
                            folder_defaults={"hr_dir": args.train_hr, "scale": args.scale,
                                             "training": True, "crop_size": args.crop_size,
                                             "repeat": args.repeat, "augment": args.augment, "seed": args.seed})
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.workers,
                         shuffle=False, seed=args.seed)
    block_count = 0
    def hr_batches():
        nonlocal block_count
        for index, batch in enumerate(loader):
            if index % 50 == 0:
                print(f"PCA batch {index + 1}/{len(loader)}", flush=True)
            _, hr = supervised_batch(batch, "cpu", scale=args.scale)
            block_count += hr.shape[0] * (hr.shape[-2] // args.scale) * (hr.shape[-1] // args.scale)
            yield hr
    decoder = BlockNullSpacePCA.fit(hr_batches(), scale=args.scale, rank=args.rank)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    decoder.save(output)
    eigenvalues = decoder.eigenvalues
    evr = eigenvalues[:, :args.rank].sum(1) / eigenvalues.sum(1).clamp_min(1e-15)
    metadata = {"config": vars(args), "manifest": dataset_manifest(dataset), "channels": decoder.channels,
                "blocks_per_channel": block_count,
                "eigenvalues": eigenvalues.tolist(), "train_explained_variance": evr.tolist(),
                "basis_sha256": sha256(output), "torch_version": str(torch.__version__),
                "coordinate_system": "Helmert null basis; pixel-shuffle block order"}
    write_json(str(output) + ".json", metadata)
    print(f"Saved basis: {output}; train EVR={evr.tolist()}", flush=True)


if __name__ == "__main__":
    main()
