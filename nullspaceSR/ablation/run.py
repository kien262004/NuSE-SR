"""Run independent local/global backbone ablations on an existing PCA basis."""
import argparse
import copy
import csv
import json
import os
import platform
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import yaml

from ..analysis.predictability import Moments, sample_indices, sampled_values
from ..analysis.research import benchmark, prefix_decoder
from ..data import DIV2KDataset, make_loader, supervised_batch
from ..engine import train_one_epoch
from ..evaluation import evaluate
from ..metrics import LPIPSMetric
from ..models import NuSESR
from ..models.ablation import AblationBackbone
from ..pca import BlockNullSpacePCA
from ..utils import (get_device, model_from_checkpoint, restore_rng, rng_state,
                     save_checkpoint, seed_everything, sha256, write_csv, write_json)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = {
    "output": "runs/ablation_study",
    "data": {"train_hr": None, "val_hr": None, "scale": 4, "methods": ["average_pool", "bicubic"],
             "crop_size": 256, "repeat": 4, "batch_size": 8, "workers": 4},
    "basis": "runs/architecture_research/basis/full.pt",
    "experiment": {"ranks": [8], "seeds": [2026], "epochs": 100, "lr": .001, "min_lr": .000001,
                   "weight_decay": 0., "grad_clip": 1., "val_every": 5,
                   "variants": [{"name": "linear", "variant": "linear"},
                                {"name": "cnn", "variant": "cnn"},
                                {"name": "g0_local", "variant": "local"},
                                {"name": "g1_global", "variant": "global"},
                                {"name": "g2_wider", "variant": "wider", "width": 17},
                                {"name": "g3_context", "variant": "context"},
                                {"name": "local_c5", "variant": "local", "corrected": 5}]},
    "model": {"width": 16, "corrected": 2, "linear_groups": "channel", "gate_hidden": 8},
    "evaluation": {"lpips": False, "shave": 0, "samples_per_image": 4096,
                   "tile_size": 64, "tile_check_images": 4},
    "runtime": {"device": "auto", "workers_seed": 2026},
    "benchmark": {"lr_size": [256, 256], "warmup": 10, "iterations": 50},
}


def read_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT):
        raise ValueError("Unknown ablation configuration section")
    cfg = copy.deepcopy(DEFAULT)
    for key, value in raw.items():
        if isinstance(cfg[key], dict):
            if not isinstance(value, dict) or set(value) - set(cfg[key]):
                raise ValueError(f"Unknown or invalid keys in {key}")
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


@torch.no_grad()
def tiled_predict(model, lr, tile_size, halo=3):
    """Run LR tiles with enough local halo; global gates still see each tile."""
    if lr.shape[0] != 1 or tile_size < 1 or halo < 3:
        raise ValueError("Tiled evaluation expects batch=1, positive tile size, halo>=3")
    h, w = lr.shape[-2:]
    scale = model.scale
    output = lr.new_empty((1, model.channels, h * scale, w * scale))
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1, x1 = min(h, y + tile_size), min(w, x + tile_size)
            ya, xa, yb, xb = max(0, y - halo), max(0, x - halo), min(h, y1 + halo), min(w, x1 + halo)
            sr = model(lr[..., ya:yb, xa:xb])
            output[..., y*scale:y1*scale, x*scale:x1*scale] = sr[..., (y-ya)*scale:(y1-ya)*scale,
                                                                       (x-xa)*scale:(x1-xa)*scale]
    return output


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output = Path(cfg["output"]).resolve()
        self.device = get_device(cfg["runtime"]["device"])
        data, exp = cfg["data"], cfg["experiment"]
        if not data["methods"] or set(data["methods"]) - {"average_pool", "bicubic", "bilinear"}:
            raise ValueError("Unsupported LR method")
        if any(type(v) is not int or v < 1 for v in (data["batch_size"], data["repeat"], exp["epochs"], exp["val_every"])):
            raise ValueError("Batch, repeat, epochs and val_every must be positive")
        if not exp["seeds"] or len(set(exp["seeds"])) != len(exp["seeds"]):
            raise ValueError("Provide distinct seeds")
        if not exp["ranks"] or len(set(exp["ranks"])) != len(exp["ranks"]):
            raise ValueError("Provide distinct ranks")
        names = [item["name"] for item in exp["variants"]]
        if not names or len(names) != len(set(names)) or any(
            not name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in name)
            for name in names):
            raise ValueError("Variant names must be distinct and contain only letters, digits, _ or -")
        basis = Path(cfg["basis"])
        if not basis.is_file():
            raise FileNotFoundError(f"Full-rank PCA basis not found: {basis}; fit rank scale²-1 on train HR")
        self.full_decoder = BlockNullSpacePCA.load(basis)
        if self.full_decoder.scale != data["scale"] or self.full_decoder.rank != data["scale"]**2 - 1:
            raise ValueError("Ablations require a full-rank PCA basis matching the configured scale")
        self.channels = self.full_decoder.channels
        if any(type(k) is not int or not 1 <= k <= self.full_decoder.rank for k in exp["ranks"]):
            raise ValueError("Rank outside the PCA basis")
        train, val = self.dataset(True, data["methods"][0], exp["seeds"][0]), self.dataset(False, data["methods"][0], 0)
        if set(train.files) & set(val.files):
            raise ValueError("Train and validation images overlap")
        def manifest(files):
            return [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]
        source = sorted((ROOT / "nullspaceSR").rglob("*.py"))
        protocol = {"config": cfg, "train": manifest(train.files), "validation": manifest(val.files),
                    "basis_sha256": sha256(basis), "sources": {str(p.relative_to(ROOT)): sha256(p) for p in source},
                    "execution": {"torch": str(torch.__version__), "device": str(self.device),
                                  "hardware": torch.cuda.get_device_name(self.device) if self.device.type == "cuda"
                                  else platform.processor() or platform.machine(), "cpu_threads": torch.get_num_threads()}}
        path = self.output / "protocol.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Ablation config, data, basis, source or hardware changed; use a new output folder")
        write_json(path, protocol)
        self.basis_hash = protocol["basis_sha256"]
        self.lpips = LPIPSMetric().to(self.device) if cfg["evaluation"]["lpips"] else None

    def dataset(self, training, method, seed):
        d = self.cfg["data"]
        return DIV2KDataset(d["train_hr" if training else "val_hr"], scale=d["scale"], training=training,
                            crop_size=d["crop_size"], repeat=d["repeat"], augment=training, seed=seed,
                            resize={"method": method, "size": None, "antialias": True})

    def run(self):
        cfg, d, exp = self.cfg, self.cfg["data"], self.cfg["experiment"]
        for method in d["methods"]:
            for rank in exp["ranks"]:
                for spec in exp["variants"]:
                    options = {**cfg["model"], **{k: v for k, v in spec.items() if k != "name"}}
                    for seed in exp["seeds"]:
                        folder = self.output / "runs" / method / f"{spec['name']}_k{rank}_seed{seed}"
                        if (folder / "DONE").is_file():
                            print(f"Skip complete: {folder}", flush=True)
                            continue
                        print(f"Run {method} K={rank} {spec['name']} seed={seed}", flush=True)
                        seed_everything(seed)
                        decoder = prefix_decoder(self.full_decoder, rank)
                        predictor = AblationBackbone(self.channels, rank, d["scale"], **options)
                        model = NuSESR(predictor, decoder).to(self.device)
                        architecture = {"name": "NuSESR", "backbone": "ablation", "null_space": True,
                                        "channels": self.channels, "rank": rank, "scale": d["scale"],
                                        "width": options["width"], "ablation": options}
                        optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=exp["lr"], weight_decay=exp["weight_decay"])
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp["epochs"], eta_min=exp["min_lr"])
                        start, best, history = 1, float("-inf"), []
                        last = folder / "last.pt"
                        if last.exists():
                            ck = torch.load(last, map_location="cpu", weights_only=True)
                            if ck["architecture"] != architecture or ck["basis_sha256"] != self.basis_hash:
                                raise ValueError("Checkpoint architecture or basis differs from protocol")
                            model.load_state_dict(ck["model"])
                            optimizer.load_state_dict(ck["optimizer"])
                            scheduler.load_state_dict(ck["scheduler"])
                            restore_rng(ck["rng"])
                            start, best, history = ck["epoch"] + 1, ck["best_psnr"], ck["history"]
                        train_data, val_data = self.dataset(True, method, seed), self.dataset(False, method, seed)
                        train_loader = make_loader(train_data, batch_size=d["batch_size"], workers=d["workers"],
                                                   shuffle=True, seed=seed, pin_memory=self.device.type == "cuda")
                        val_loader = make_loader(val_data, batch_size=1, workers=d["workers"], pin_memory=self.device.type == "cuda")
                        for epoch in range(start, exp["epochs"] + 1):
                            train_data.set_epoch(epoch)
                            train_loader.generator.manual_seed(seed + epoch)
                            metrics = train_one_epoch(model, train_loader, optimizer, self.device,
                                                      grad_clip=exp["grad_clip"], epoch=epoch, log_every=100)
                            record = {"epoch": epoch, **metrics}
                            improved = False
                            if epoch % exp["val_every"] == 0 or epoch == exp["epochs"]:
                                summary, _ = evaluate(model, val_loader, self.device, shave=cfg["evaluation"]["shave"],
                                                      lpips_metric=self.lpips)
                                record["validation"] = summary
                                improved = summary["rgb_psnr"] > best
                                best = max(best, summary["rgb_psnr"])
                            history.append(record)
                            scheduler.step()
                            state = {"format_version": 2, "epoch": epoch, "architecture": architecture,
                                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                     "scheduler": scheduler.state_dict(), "rng": rng_state(), "best_psnr": best,
                                     "history": history, "basis_sha256": self.basis_hash}
                            if improved:
                                save_checkpoint(folder / "best.pt", state)
                            save_checkpoint(last, state)
                            write_json(folder / "history.json", history)
                            print(f"epoch {epoch}/{exp['epochs']} loss={metrics[model.loss_name]:.6f}"
                                  + (f" val_psnr={record['validation']['rgb_psnr']:.4f}" if "validation" in record else ""), flush=True)
                        ck = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                        model = model_from_checkpoint(ck).to(self.device).eval()
                        summary, rows = evaluate(model, val_loader, self.device, shave=cfg["evaluation"]["shave"],
                                                 lpips_metric=self.lpips)
                        write_json(folder / "quality.json", summary)
                        write_csv(folder / "per_image.csv", rows)
                        self.components_and_tiles(model, val_loader, folder, seed)
                        write_json(folder / "benchmark.json", benchmark(model, cfg, self.device))
                        write_json(folder / "DONE", {"epoch": ck["epoch"], "completed_at": time.time()})
        self.report()

    @torch.no_grad()
    def components_and_tiles(self, model, loader, folder, seed):
        cfg = self.cfg["evaluation"]
        moments = Moments(model.channels * model.decoder.rank, self.device)
        generator = torch.Generator().manual_seed(self.cfg["runtime"]["workers_seed"] + 100000)
        rows = []
        for index, batch in enumerate(loader):
            lr, hr = supervised_batch(batch, self.device, channels=model.channels, scale=model.scale)
            target, prediction = model.decoder.encode_hr(hr), model.predict_coefficients(lr)
            indices = sample_indices(1, *lr.shape[-2:], cfg["samples_per_image"], generator)
            moments.update(sampled_values(target, indices), sampled_values(prediction, indices))
            if index < cfg["tile_check_images"]:
                whole = model(lr)
                tiled = tiled_predict(model, lr, cfg["tile_size"])
                rows.append({"name": batch["id"][0], "tile_size_lr": cfg["tile_size"], "halo_lr": 3,
                             "full_vs_tiled_mse": (whole.double() - tiled.double()).square().mean().item(),
                             "full_vs_tiled_max_abs": (whole - tiled).abs().max().item(),
                             "tiled_dc_max_abs": (model.operator(tiled) - lr).abs().max().item()})
        # Training mean is not available for this dataset protocol; report zero baseline explicitly.
        values = moments.rows(model.channels, model.decoder.rank, torch.zeros(model.channels * model.decoder.rank))
        for row in values:
            row.pop("train_mean_baseline_mse")
            row.pop("gain_vs_train_mean")
        write_csv(folder / "components.csv", values)
        if rows:
            write_csv(folder / "tiles.csv", rows)

    def report(self):
        results, by_group = [], {}
        for done in sorted((self.output / "runs").glob("*/*/DONE")):
            folder = done.parent
            ck = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            q = json.loads((folder / "quality.json").read_text())
            cost = json.loads((folder / "benchmark.json").read_text())
            prefix, seed_text = folder.name.rsplit("_seed", 1)
            name, rank_text = prefix.rsplit("_k", 1)
            row = {"method": folder.parent.name, "variant": name, "rank": int(rank_text),
                   "seed": int(seed_text), "best_epoch": ck["epoch"], **q, **cost}
            results.append(row)
            by_group.setdefault((row["method"], name, row["rank"]), []).append(row)
        if not results:
            print("No completed ablation runs to report", flush=True)
            return
        out = self.output / "reports"
        fields = list(dict.fromkeys(k for row in results for k in row))
        write_csv(out / "runs.csv", [{k: row.get(k) for k in fields} for row in results])
        summaries = []
        for (method, name, rank), rows in by_group.items():
            item = {"method": method, "variant": name, "rank": rank, "seeds": len(rows)}
            for key in ("rgb_psnr", "rgb_ssim", "lpips", "rgb_mse", "coefficient_image_mse", "end_to_end_ms", "params"):
                if key in rows[0]:
                    vals = [r[key] for r in rows]
                    item[key + "_mean"] = statistics.fmean(vals)
                    item[key + "_std"] = statistics.pstdev(vals)
            summaries.append(item)
        fields = list(dict.fromkeys(k for row in summaries for k in row))
        write_csv(out / "summary.csv", [{k: row.get(k) for k in fields} for row in summaries])
        pairs = []
        index = {(r["method"], r["rank"], r["variant"], r["seed"]): r for r in results}
        for method, rank in sorted({(r["method"], r["rank"]) for r in results}):
            for variant in sorted({r["variant"] for r in results if r["method"] == method and r["rank"] == rank} - {"g0_local"}):
                paired = []
                for seed in self.cfg["experiment"]["seeds"]:
                    if (method, rank, variant, seed) not in index or (method, rank, "g0_local", seed) not in index:
                        continue
                    def image_psnr(name):
                        path = self.output / "runs" / method / f"{name}_k{rank}_seed{seed}" / "per_image.csv"
                        with path.open(encoding="utf-8") as stream:
                            return {r["name"]: float(r["rgb_psnr"]) for r in csv.DictReader(stream)}
                    a, b = image_psnr(variant), image_psnr("g0_local")
                    if a.keys() != b.keys():
                        raise ValueError("Paired runs have different validation image IDs")
                    paired.append(np.array([a[k] - b[k] for k in sorted(a)]))
                if paired:
                    per_image = np.stack(paired).mean(0)
                    rng = np.random.default_rng(2026)
                    bootstrap = per_image[rng.integers(0, len(per_image), (10000, len(per_image)))].mean(1)
                    pairs.append({"method": method, "rank": rank, "variant": variant, "baseline": "g0_local",
                                  "paired_seeds": len(paired), "images": len(per_image),
                                  "delta_psnr": float(per_image.mean()),
                                  "ci_image_95_low": float(np.quantile(bootstrap, .025)),
                                  "ci_image_95_high": float(np.quantile(bootstrap, .975)),
                                  "note": "paired image bootstrap conditional on trained seeds; not seed uncertainty"})
        if pairs:
            write_csv(out / "paired_vs_local.csv", pairs)
        write_json(out / "index.json", {"completed_runs": len(results), "seed_groups": len(summaries),
                   "selection": "best validation PSNR; paired CI is conditional on the trained models",
                   "warning": "No held-out test set is evaluated; compare actual params and latency for matched-cost controls."})
        print(f"Ablation report: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ablation_study.yaml"))
    parser.add_argument("--stage", choices=("run", "report"), default="run")
    for name in ("train-hr", "val-hr", "basis", "output", "device"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--smoke", action="store_true", help="Tiny code check, not a research result")
    args = parser.parse_args()
    cfg = read_config(args.config)
    for key in ("train_hr", "val_hr"):
        if getattr(args, key):
            cfg["data"][key] = getattr(args, key)
        if not cfg["data"][key]:
            parser.error(f"data.{key} is required")
        cfg["data"][key] = str(Path(cfg["data"][key]).resolve())
    for key in ("basis", "output"):
        cfg[key] = str(Path(getattr(args, key) or cfg[key]).resolve())
    if args.device:
        cfg["runtime"]["device"] = args.device
    if args.smoke:
        cfg["data"].update(crop_size=cfg["data"]["scale"] * 8, repeat=1, batch_size=1, workers=0)
        cfg["experiment"].update(ranks=[min(3, cfg["data"]["scale"]**2-1)], seeds=[2026], epochs=2, val_every=1,
                                  variants=[{"name": "g0_local", "variant": "local"},
                                            {"name": "g1_global", "variant": "global"},
                                            {"name": "g3_context", "variant": "context"}])
        cfg["model"].update(width=4, corrected=2, gate_hidden=2)
        cfg["evaluation"].update(lpips=False, tile_size=4, tile_check_images=1, samples_per_image=16)
        cfg["benchmark"].update(lr_size=[8, 8], warmup=1, iterations=2)
    runner = Runner(cfg)
    runner.run() if args.stage == "run" else runner.report()


if __name__ == "__main__":
    main()
