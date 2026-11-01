"""Server runner for oracle, context predictability and backbone experiments."""
import argparse
import copy
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import yaml

from ..data import DIV2KDataset, make_loader, supervised_batch
from ..losses import coefficient_image_mse
from ..metrics import LPIPSMetric, rgb_mse, rgb_psnr, rgb_ssim
from ..models import NuSESR
from ..operators import AveragePoolSR
from ..pca import BlockNullSpacePCA
from ..utils import (get_device, model_from_checkpoint, restore_rng, rng_state, save_checkpoint,
                     seed_everything, sha256, write_csv, write_json)
from .predictability import ContextCNN, Moments, StreamingRidge, sample_indices, sampled_values


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "output": "runs/architecture_research",
    "data": {"train_hr": None, "val_hr": None, "scale": 4, "methods": ["average_pool", "bicubic"],
             "crop_size": 256, "repeat": 4, "batch_size": 8, "workers": 4},
    "pca": {"basis": None, "repeat": 4},
    "analysis": {"ranks": [1, 2, 4, 8, 12, 15], "contexts": [1, 3, 5, 9, 13],
                 "channel_modes": ["same_channel", "rgb"], "train_samples_per_crop": 512,
                 "val_samples_per_image": 4096, "ridge_alpha": 0.0001},
    "cnn": {"enabled": True, "contexts": [1, 3, 5, 9, 13], "width": 16,
            "epochs": 20, "seeds": [2026], "lr": .001, "min_lr": .000001, "grad_clip": 1.0},
    "evaluation": {"lpips": False, "shave": 0, "max_val_images": None},
    "runtime": {"device": "auto", "seed": 2026, "max_train_batches": None},
    "benchmark": {"lr_size": [256, 256], "warmup": 10, "iterations": 50},
    "sweep": {"ranks": [4, 8, 15], "seeds": [2026, 2027, 2028], "epochs": 100,
              "variants": [{"name": "nuse_w16_n2", "backbone": "nuse", "width": 16, "num_blocks": 2, "block": "plain"},
                           {"name": "nuse_w32_n7", "backbone": "nuse", "width": 32, "num_blocks": 7, "block": "plain"},
                           {"name": "nuse_rep_w32_n7", "backbone": "nuse", "width": 32, "num_blocks": 7, "block": "rep"},
                           {"name": "imdn_w64", "backbone": "imdn", "width": 64}]},
}


def read_config(path):
    supplied = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(supplied, dict):
        raise ValueError("Research config must be a mapping")
    config = copy.deepcopy(DEFAULTS)
    for key, value in supplied.items():
        if key not in config:
            raise ValueError(f"Unknown research key: {key}")
        if isinstance(config[key], dict):
            if not isinstance(value, dict) or set(value) - set(config[key]):
                raise ValueError(f"Invalid keys/mapping in {key}")
            config[key].update(value)
        else:
            config[key] = value
    return config


def prefix_decoder(decoder, rank):
    """All ranks share one fitted coordinate system and training mean."""
    return BlockNullSpacePCA(decoder.scale, decoder.mean, decoder.components[..., :rank],
                              decoder.eigenvalues if decoder.eigenvalues.numel() else None)


def average_rows(rows, keys):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for group, values in grouped.items():
        entry = dict(zip(keys, group))
        entry["images"] = len(values)
        for key in values[0]:
            if key not in keys and isinstance(values[0][key], (float, int)):
                entry[key] = (max(v[key] for v in values) if key in {"dc_max_abs", "decomposition_abs"}
                              else statistics.fmean(v[key] for v in values))
        result.append(entry)
    return result


def benchmark(model, config, device):
    """Batch-one fp32 wall latency, synchronized on CUDA; no transfer/preprocessing."""
    cfg = config["benchmark"]
    model.eval()
    x = torch.rand(1, model.channels, *cfg["lr_size"], device=device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def measure(fn):
        with torch.no_grad():
            for _ in range(cfg["warmup"]):
                fn()
            times = []
            for _ in range(cfg["iterations"]):
                sync()
                begin = time.perf_counter()
                fn()
                sync()
                times.append((time.perf_counter() - begin) * 1000)
        return statistics.median(times), statistics.pstdev(times)
    with torch.no_grad():
        coeffs = model.predictor(x)
    predictor_ms, _ = measure(lambda: model.predictor(x))
    decoder_ms, _ = measure(lambda: model.operator.pseudoinverse(x) + model.decoder(coeffs))
    baseline_bytes = None
    if device.type == "cuda":
        baseline_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    total_ms, spread = measure(lambda: model(x))
    macs = [0]
    def hook(module, inputs, output):
        macs[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
    handles = [m.register_forward_hook(hook) for m in model.predictor.modules() if isinstance(m, torch.nn.Conv2d)]
    try:
        with torch.no_grad():
            model.predictor(x)
    finally:
        for handle in handles:
            handle.remove()
    decoder_macs = x.shape[-2] * x.shape[-1] * model.channels * model.scale**2 * model.decoder.rank
    return {"params": sum(p.numel() for p in model.parameters()),
            "model_bytes": sum(p.numel() * p.element_size() for p in [*model.parameters(), *model.buffers()]),
            "predictor_macs": macs[0], "decoder_macs": decoder_macs,
            "total_macs": macs[0] + decoder_macs,
            "predictor_ms": predictor_ms, "decoder_ms": decoder_ms,
            "end_to_end_ms": total_ms, "latency_std_ms": spread,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "cuda_peak_extra_bytes": (max(0, torch.cuda.max_memory_allocated(device) - baseline_bytes)
                                      if device.type == "cuda" else None),
            "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "precision": "fp32", "batch_size": 1, "lr_height": x.shape[-2], "lr_width": x.shape[-1],
            "protocol": "synchronized wall time; excludes H2D/preprocessing; conv MACs + decoder synthesis; excludes adds/activations"}


class ResearchRunner:
    def __init__(self, config):
        self.config = config
        self.output = Path(config["output"]).resolve()
        self.device = get_device(config["runtime"]["device"])
        self.seed = config["runtime"]["seed"]
        seed_everything(self.seed)
        data, analysis = config["data"], config["analysis"]
        self.scale, self.rank = data["scale"], data["scale"]**2 - 1
        self.ranks = analysis["ranks"]
        if (not self.ranks or len(set(self.ranks)) != len(self.ranks)
                or any(type(k) is not int or not 1 <= k <= self.rank for k in self.ranks)):
            raise ValueError(f"analysis.ranks must be distinct integers between 1 and {self.rank}")
        for contexts in (analysis["contexts"], config["cnn"]["contexts"]):
            if not contexts or any(type(r) is not int or r < 1 or r % 2 == 0 for r in contexts):
                raise ValueError("contexts must contain positive odd integers")
        if not data["methods"] or set(data["methods"]) - {"average_pool", "bicubic", "bilinear"}:
            raise ValueError("methods must select average_pool, bicubic or bilinear")
        if not analysis["channel_modes"] or set(analysis["channel_modes"]) - {"same_channel", "rgb"}:
            raise ValueError("channel_modes must select same_channel and/or rgb")
        for value in (analysis["train_samples_per_crop"], analysis["val_samples_per_image"],
                      config["cnn"]["epochs"], config["benchmark"]["iterations"]):
            if type(value) is not int or value < 1:
                raise ValueError("Sample counts, epochs and benchmark iterations must be positive integers")
        if analysis["ridge_alpha"] <= 0 or config["benchmark"]["warmup"] < 0:
            raise ValueError("ridge_alpha must be positive; warmup must be nonnegative")
        self.train = self.dataset(True, data["methods"][0])
        self.val = self.dataset(False, data["methods"][0])
        if set(self.train.files) & set(self.val.files):
            raise ValueError("Train and validation image paths overlap")
        def files(paths):
            return [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]
        source_files = [Path(__file__), Path(__file__).with_name("predictability.py"),
                        ROOT / "nullspaceSR/pca.py", ROOT / "nullspaceSR/metrics.py",
                        ROOT / "nullspaceSR/models/nuse.py", ROOT / "train.py"]
        protocol = {"config": config, "train": files(self.train.files), "validation": files(self.val.files),
                    "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
                    "execution": {"torch": str(torch.__version__), "device": str(self.device),
                                  "hardware": torch.cuda.get_device_name(self.device) if self.device.type == "cuda"
                                  else platform.processor() or platform.machine(), "cpu_threads": torch.get_num_threads()},
                    "external_basis_sha256": sha256(config["pca"]["basis"]) if config["pca"]["basis"] else None}
        old = self.output / "protocol.json"
        if old.exists() and json.loads(old.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Research protocol/data/code changed; choose a new output directory")
        write_json(old, protocol)
        write_json(self.output / "environment.json", {"torch": str(torch.__version__), "python": sys.version,
                                                     "device": str(self.device)})
        self.decoder = None
        self.lpips = None

    def dataset(self, training, method, repeat=None):
        cfg = self.config["data"]
        return DIV2KDataset(cfg["train_hr" if training else "val_hr"], scale=cfg["scale"],
                            training=training, crop_size=cfg["crop_size"],
                            repeat=repeat if repeat is not None else cfg["repeat"],
                            augment=training, seed=self.config["runtime"]["seed"],
                            resize={"method": method, "size": None, "antialias": True})

    def batches(self, training, method, epoch=0, repeat=None, seed=None):
        data = self.dataset(training, method, repeat)
        data.set_epoch(epoch)
        loader = make_loader(data, batch_size=self.config["data"]["batch_size"] if training else 1,
                             workers=self.config["data"]["workers"], shuffle=training,
                             seed=(self.seed if seed is None else seed) + epoch,
                             pin_memory=self.device.type == "cuda")
        limit = self.config["runtime"]["max_train_batches"] if training else self.config["evaluation"]["max_val_images"]
        for step, batch in enumerate(loader, 1):
            if limit is not None and step > limit:
                break
            yield step, batch

    def ensure_basis(self):
        path = self.output / "basis/full.pt"
        if not path.exists():
            external = self.config["pca"]["basis"]
            if external:
                decoder = BlockNullSpacePCA.load(external)
            else:
                print("Fitting one full-rank PCA basis from TRAIN HR only", flush=True)
                def images():
                    for step, batch in self.batches(True, self.config["data"]["methods"][0],
                                                     repeat=self.config["pca"]["repeat"]):
                        if step % 25 == 0:
                            print(f"PCA batch {step}", flush=True)
                        yield batch["target"]
                decoder = BlockNullSpacePCA.fit(images(), self.scale, self.rank)
            if decoder.scale != self.scale or decoder.rank != self.rank or decoder.channels != 3:
                raise ValueError("Research basis must be full rank (scale²-1), RGB and match scale")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            decoder.save(temporary)
            os.replace(temporary, path)
        fingerprint = self.output / "basis/fingerprint.json"
        actual = {"sha256": sha256(path)}
        if fingerprint.exists() and json.loads(fingerprint.read_text()) != actual:
            raise ValueError("Cached research basis was modified; use a new output directory")
        write_json(fingerprint, actual)
        self.decoder = BlockNullSpacePCA.load(path).to(self.device)
        self.operator = AveragePoolSR(self.scale)
        if self.config["evaluation"]["lpips"]:
            self.lpips = LPIPSMetric().to(self.device)
        stats_path = self.output / "basis/train_coefficients.pt"
        if not stats_path.exists():
            moments = Moments(3 * self.rank, self.device)
            with torch.no_grad():
                for _, batch in self.batches(True, self.config["data"]["methods"][0]):
                    target = self.decoder.encode_hr(batch["target"].to(self.device))
                    moments.update(target.flatten(2).transpose(1, 2).flatten(0, 1))
            save_checkpoint(stats_path, moments.state())
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        self.train_mean = stats["total"] / stats["count"]

    @torch.no_grad()
    def evaluate(self, predictor, method, folder):
        """Full-image quality; sampled per-component diagnostics with a fixed seed."""
        folder = Path(folder)
        moments = Moments(3 * self.rank, self.device)
        generator = torch.Generator().manual_seed(self.seed + 100000)
        decoders = {k: prefix_decoder(self.decoder, k).to(self.device) for k in self.ranks}
        rows = []
        if predictor is not None:
            predictor.eval()
        for step, batch in self.batches(False, method):
            lr, hr = supervised_batch(batch, self.device, channels=3, scale=self.scale)
            target = self.decoder.encode_hr(hr)
            predicted = target if predictor is None else predictor(lr)
            if predicted.shape != target.shape or not torch.isfinite(predicted).all():
                raise ValueError("Invalid coefficient predictor output")
            indices = sample_indices(1, *lr.shape[-2:], self.config["analysis"]["val_samples_per_image"], generator)
            moments.update(sampled_values(target, indices), sampled_values(predicted, indices))
            reference = target.reshape(1, 3, self.rank, *lr.shape[-2:])
            estimates = predicted.reshape_as(reference)
            base = self.operator.pseudoinverse(lr)
            matching_base = self.operator.pseudoinverse(self.operator(hr))
            mismatch = rgb_mse(self.operator(hr), lr).item()
            for rank, decoder in decoders.items():
                cstar = reference[:, :, :rank].flatten(1, 2)
                chat = estimates[:, :, :rank].flatten(1, 2)
                oracle = base + decoder(cstar)
                sr = base + decoder(chat)
                if not torch.isfinite(sr).all():
                    raise FloatingPointError("Non-finite research SR")
                full_mse = rgb_mse(sr, hr).item()
                residual = rgb_mse(oracle, hr).item()
                error = coefficient_image_mse(chat.double(), cstar.double(), channels=3, scale=self.scale).item()
                row = {"method": method, "rank": rank, "name": batch["id"][0],
                       "rgb_psnr": rgb_psnr(sr, hr, self.config["evaluation"]["shave"]).item(),
                       "rgb_ssim": rgb_ssim(sr, hr, self.config["evaluation"]["shave"]).item(),
                       "rgb_mse": rgb_mse(sr, hr, self.config["evaluation"]["shave"]).item(),
                       "full_image_mse": full_mse, "truncation_mse": residual,
                       "pca_truncation_mse": rgb_mse(matching_base + decoder(cstar), hr).item(),
                       "measurement_mismatch_mse": mismatch, "coefficient_image_mse": error,
                       "dc_max_abs": (self.operator(sr) - lr).abs().max().item(),
                       "decomposition_abs": abs(full_mse - residual - error)}
                if self.lpips is not None:
                    row["lpips"] = self.lpips(sr, hr, self.config["evaluation"]["shave"]).item()
                rows.append(row)
            if step % 10 == 0:
                print(f"evaluate {method}: {step} images", flush=True)
        if not rows:
            raise ValueError("No validation images evaluated")
        summary = average_rows(rows, ("method", "rank"))
        components = moments.rows(3, self.rank, self.train_mean)
        write_csv(folder / "per_image.csv", rows)
        write_csv(folder / "quality.csv", summary)
        write_csv(folder / "components.csv", components)
        write_json(folder / "summary.json", {"quality": summary, "shave": self.config["evaluation"]["shave"],
                    "component_protocol": "sample-weighted; deterministic uniform samples per image; zero-based RGB channel",
                    "quality_protocol": "full images; image mean; raw PSNR/SSIM; fixed average-pool model",
                    "basis_sha256": sha256(self.output / "basis/full.pt")})
        if predictor is not None:
            model = NuSESR(predictor, self.decoder)
            write_json(folder / "benchmark.json", benchmark(model, self.config, self.device))
        return summary

    def oracle(self):
        for method in self.config["data"]["methods"]:
            folder = self.output / "oracle" / method
            if not (folder / "DONE").exists():
                self.evaluate(None, method, folder)
                write_json(folder / "DONE", {"complete": True})

    def ridge(self):
        cfg = self.config["analysis"]
        for method in self.config["data"]["methods"]:
            for mode in cfg["channel_modes"]:
                for context in cfg["contexts"]:
                    folder = self.output / "ridge" / method / f"{mode}_r{context}"
                    if (folder / "DONE").exists():
                        continue
                    print(f"Ridge: {method}, {mode}, context {context}", flush=True)
                    system = StreamingRidge(3, self.rank, context, mode, self.device)
                    generator = torch.Generator().manual_seed(self.seed)
                    with torch.no_grad():
                        for step, batch in self.batches(True, method):
                            lr, hr = supervised_batch(batch, self.device, channels=3, scale=self.scale)
                            indices = sample_indices(lr.shape[0], *lr.shape[-2:], cfg["train_samples_per_crop"], generator)
                            system.update(lr, self.decoder.encode_hr(hr), indices)
                            if step % 25 == 0:
                                print(f"  ridge batch {step}; sampled positions {system.count}", flush=True)
                    predictor = system.solve(cfg["ridge_alpha"]).to(self.device)
                    del system
                    save_checkpoint(folder / "ridge.pt", {"model": predictor.state_dict(), "context": context,
                        "channel_mode": mode, "rank": self.rank, "alpha": cfg["ridge_alpha"],
                        "basis_sha256": sha256(self.output / "basis/full.pt")})
                    self.evaluate(predictor, method, folder)
                    write_json(folder / "DONE", {"complete": True})

    def cnn(self):
        cfg = self.config["cnn"]
        if not cfg["enabled"]:
            return
        for method in self.config["data"]["methods"]:
            for context in cfg["contexts"]:
                for seed in cfg["seeds"]:
                    folder = self.output / "cnn" / method / f"rgb_r{context}_seed{seed}"
                    if (folder / "DONE").exists():
                        continue
                    seed_everything(seed)
                    predictor = ContextCNN(3, self.rank, context, cfg["width"]).to(self.device)
                    optimizer = torch.optim.AdamW(predictor.parameters(), lr=cfg["lr"], weight_decay=0)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=cfg["min_lr"])
                    start, history = 1, []
                    if (folder / "last.pt").exists():
                        checkpoint = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
                        predictor.load_state_dict(checkpoint["model"])
                        optimizer.load_state_dict(checkpoint["optimizer"])
                        scheduler.load_state_dict(checkpoint["scheduler"])
                        restore_rng(checkpoint["rng"])
                        start, history = checkpoint["epoch"] + 1, checkpoint["history"]
                    for epoch in range(start, cfg["epochs"] + 1):
                        predictor.train()
                        total, pixels = 0.0, 0
                        for step, batch in self.batches(True, method, epoch=epoch, seed=seed):
                            lr, hr = supervised_batch(batch, self.device, channels=3, scale=self.scale)
                            with torch.no_grad():
                                target = self.decoder.encode_hr(hr)
                            loss = coefficient_image_mse(predictor(lr), target, channels=3, scale=self.scale)
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Non-finite CNN loss")
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(predictor.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
                            optimizer.step()
                            total += loss.item() * hr.numel()
                            pixels += hr.numel()
                            if step % 25 == 0:
                                print(f"  CNN {method} r={context} epoch={epoch} batch={step} loss={loss.item():.6f}", flush=True)
                        if not pixels:
                            raise ValueError("Empty CNN training loader")
                        scheduler.step()
                        history.append({"epoch": epoch, "train_coefficient_image_mse": total / pixels})
                        save_checkpoint(folder / "last.pt", {"epoch": epoch, "model": predictor.state_dict(),
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(),
                            "history": history, "context": context, "width": cfg["width"], "rank": self.rank,
                            "basis_sha256": sha256(self.output / "basis/full.pt")})
                        write_json(folder / "history.json", history)
                        print(f"CNN {method} r={context} seed={seed} epoch={epoch}/{cfg['epochs']} loss={total / pixels:.6f}", flush=True)
                    self.evaluate(predictor, method, folder)
                    write_json(folder / "DONE", {"complete": True})

    def sweep(self):
        """Independent deployed models per rank; uses the existing training CLI."""
        cfg, data = self.config["sweep"], self.config["data"]
        for rank in cfg["ranks"]:
            if type(rank) is not int or not 1 <= rank <= self.rank:
                raise ValueError("sweep ranks must fit within the full PCA basis")
            basis = self.output / f"basis/rank_{rank}.pt"
            prefix_decoder(self.decoder, rank).cpu().save(basis)
            for method in data["methods"]:
                for variant in cfg["variants"]:
                    name = variant["name"]
                    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
                        raise ValueError("Variant names must use letters, numbers, underscores or hyphens")
                    for seed in cfg["seeds"]:
                        folder = self.output / "sweep" / method / f"{name}_k{rank}_seed{seed}"
                        if (folder / "DONE").exists():
                            continue
                        run_config = {"model": {k: v for k, v in variant.items() if k != "name"},
                            "data": {"train_hr": data["train_hr"], "val_hr": data["val_hr"],
                                     "crop_size": data["crop_size"], "repeat": data["repeat"], "workers": data["workers"],
                                     "resize": {"method": method, "size": None, "antialias": True}},
                            "train": {"output": str(folder), "epochs": cfg["epochs"], "batch_size": data["batch_size"]},
                            "runtime": {"device": str(self.device), "seed": seed},
                            "evaluation": {"lpips": self.config["evaluation"]["lpips"], "shave": self.config["evaluation"]["shave"]},
                            "logging": {"enabled": False}}
                        run_config["model"].update(null_space=True, scale=self.scale, rank=rank, basis=str(basis))
                        path = folder / "config.yaml"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(yaml.safe_dump(run_config), encoding="utf-8")
                        command = [sys.executable, str(ROOT / "train.py"), "--config", str(path)]
                        if (folder / "last.pt").exists():
                            command += ["--resume", str(folder / "last.pt")]
                        print("Running:", " ".join(command), flush=True)
                        with (folder / "console.log").open("a", encoding="utf-8") as log:
                            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                       text=True, encoding="utf-8", errors="replace")
                            for line in process.stdout:
                                print(line, end="", flush=True)
                                log.write(line)
                                log.flush()
                            if process.wait():
                                raise RuntimeError(f"Backbone run failed; see {folder / 'console.log'}")
                        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                        model = model_from_checkpoint(checkpoint).to(self.device).eval()
                        train_params = sum(p.numel() for p in model.parameters())
                        if hasattr(model.predictor, "fuse"):
                            model.predictor.fuse()
                        cost = benchmark(model, self.config, self.device)
                        cost["training_params"] = train_params
                        write_json(folder / "benchmark.json", cost)
                        selected = next(row["validation"] for row in checkpoint["history"] if row["epoch"] == checkpoint["epoch"])
                        write_json(folder / "selected_quality.json", selected)
                        write_json(folder / "DONE", {"complete": True})

    def report(self):
        output = self.output / "reports"
        components, quality, costs, sweep = [], [], [], []
        import csv
        for path in sorted(self.output.glob("**/components.csv")):
            if not path.with_name("DONE").exists():
                continue
            relative = path.relative_to(self.output).parts
            stage, method = relative[:2]
            name = relative[2] if stage != "oracle" else "oracle"
            context = int(name.split("_r")[-1].split("_")[0]) if "_r" in name else 0
            metadata = {"predictor": stage, "method": method, "run": name, "context": context,
                        "seed": int(name.rsplit("_seed", 1)[1]) if "_seed" in name else None,
                        "channel_mode": "same_channel" if name.startswith("same_channel") else "rgb"}
            with path.open(encoding="utf-8") as stream:
                components.extend({**metadata, **row} for row in csv.DictReader(stream))
            with path.with_name("quality.csv").open(encoding="utf-8") as stream:
                quality.extend({**metadata, **row} for row in csv.DictReader(stream))
            if path.with_name("benchmark.json").exists():
                costs.append({**metadata, **json.loads(path.with_name("benchmark.json").read_text())})
        for path in sorted(self.output.glob("sweep/**/selected_quality.json")):
            if not path.with_name("DONE").exists():
                continue
            cost = json.loads(path.with_name("benchmark.json").read_text())
            name, suffix = path.parent.name.rsplit("_k", 1)
            rank, seed = suffix.split("_seed")
            sweep.append({"method": path.relative_to(self.output).parts[1], "run": path.parent.name,
                          "variant": name, "rank": int(rank), "seed": int(seed),
                          **json.loads(path.read_text()), **cost})
        for name, rows in (("r2_vs_context.csv", components), ("quality_vs_rank.csv", quality),
                           ("cost.csv", costs), ("backbone_sweep.csv", sweep)):
            if rows:
                keys = list(dict.fromkeys(key for row in rows for key in row))
                write_csv(output / name, [{key: row.get(key) for key in keys} for row in rows])
        oracle_quality = [r for r in quality if r["predictor"] == "oracle"]
        oracle_components = [r for r in components if r["predictor"] == "oracle"]
        if oracle_quality:
            write_csv(output / "oracle_vs_rank.csv", oracle_quality)
        if oracle_components:
            write_csv(output / "coefficient_energy.csv", oracle_components)
        sweep_summary = []
        groups = {}
        for row in sweep:
            groups.setdefault((row["method"], row["variant"], row["rank"]), []).append(row)
        for (method, variant, rank), rows in groups.items():
            summary = {"method": method, "variant": variant, "rank": rank, "seeds": len(rows)}
            for key in ("rgb_psnr", "rgb_ssim", "lpips", "end_to_end_ms", "params", "total_macs"):
                if key in rows[0]:
                    values = [row[key] for row in rows]
                    summary[f"{key}_mean"] = statistics.fmean(values)
                    summary[f"{key}_std"] = statistics.pstdev(values)
            sweep_summary.append(summary)
        for row in sweep_summary:
            row["pareto_psnr_latency"] = not any(
                other["method"] == row["method"] and
                other["rgb_psnr_mean"] >= row["rgb_psnr_mean"] and
                other["end_to_end_ms_mean"] <= row["end_to_end_ms_mean"] and
                (other["rgb_psnr_mean"] > row["rgb_psnr_mean"] or
                 other["end_to_end_ms_mean"] < row["end_to_end_ms_mean"])
                for other in sweep_summary)
        if sweep_summary:
            keys = list(dict.fromkeys(key for row in sweep_summary for key in row))
            write_csv(output / "backbone_summary.csv", [{key: row.get(key) for key in keys} for row in sweep_summary])
        write_json(output / "index.json", {"component_rows": len(components), "quality_rows": len(quality),
                    "sweep_runs": len(sweep), "note": "Probe models predict full rank once; prefix quality is not rank-scaled deployment cost. Sweep trains independent models per rank."})
        self.plots(components, quality, output, sweep_summary)
        print(f"Reports: {output}", flush=True)

    @staticmethod
    def plots(components, quality, output, sweep_summary):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print('CSV reports ready; install ".[research]" for PNG plots', flush=True)
            return
        for method in sorted({row["method"] for row in components}):
            for kind in sorted({row["predictor"] for row in components if row["predictor"] != "oracle"}):
                for mode in ("rgb", "same_channel"):
                    subset = [r for r in components if r["method"] == method and r["predictor"] == kind and r["channel_mode"] == mode]
                    if not subset:
                        continue
                    contexts = sorted({int(r["context"]) for r in subset})
                    ranks = sorted({int(r["component"]) for r in subset})
                    fig, axes = plt.subplots(1, 3, figsize=(13, 5), squeeze=False)
                    for channel, ax in enumerate(axes[0]):
                        matrix = []
                        for rank in ranks:
                            values = []
                            for context in contexts:
                                found = [float(r["r2"]) for r in subset if int(r["channel"]) == channel and
                                         int(r["component"]) == rank and int(r["context"]) == context and r["r2"] not in ("", None)]
                                values.append(statistics.fmean(found) if found else float("nan"))
                            matrix.append(values)
                        image = ax.imshow(matrix, aspect="auto", vmin=-.2, vmax=1, cmap="viridis")
                        ax.set_xticks(range(len(contexts)), contexts)
                        ax.set_yticks(range(len(ranks)), ranks)
                        ax.set(xlabel="LR context", ylabel="PCA component", title="RGB"[channel])
                    fig.colorbar(image, ax=list(axes[0]), label="R² (clipped color scale)")
                    fig.suptitle(f"{method}: {kind}, {mode}; seed mean where applicable")
                    fig.savefig(output / f"r2_{method}_{kind}_{mode}.png", dpi=160, bbox_inches="tight")
                    plt.close(fig)
        oracle = [r for r in quality if r["predictor"] == "oracle"]
        if oracle:
            fig, ax = plt.subplots(figsize=(7, 4))
            for method in sorted({r["method"] for r in oracle}):
                rows = sorted((r for r in oracle if r["method"] == method), key=lambda r: int(r["rank"]))
                ax.plot([int(r["rank"]) for r in rows], [float(r["rgb_psnr"]) for r in rows], marker="o", label=method)
            ax.set(xlabel="PCA rank", ylabel="Oracle RGB PSNR (dB)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / "oracle_vs_rank.png", dpi=160)
            plt.close(fig)
        for method in sorted({row["method"] for row in sweep_summary}):
            fig, ax = plt.subplots(figsize=(10, 6))
            for row in sweep_summary:
                if row["method"] != method:
                    continue
                x, y = row["end_to_end_ms_mean"], row["rgb_psnr_mean"]
                ax.errorbar(x, y, xerr=row["end_to_end_ms_std"], yerr=row["rgb_psnr_std"],
                            fmt="o", color="tab:red" if row["pareto_psnr_latency"] else "tab:blue")
                ax.annotate(f"{row['variant']} K={row['rank']}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
            ax.set(xlabel="End-to-end latency (ms)", ylabel="Validation RGB PSNR (dB)", title=f"{method}: seed mean ± population std; red = Pareto")
            fig.tight_layout()
            fig.savefig(output / f"pareto_{method}.png", dpi=160)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/research_architecture.yaml"))
    parser.add_argument("--stage", choices=("all", "basis", "oracle", "ridge", "cnn", "sweep", "report"), default="all")
    for name in ("train-hr", "val-hr", "output", "device"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--smoke", action="store_true", help="Tiny execution check; not a research result")
    args = parser.parse_args()
    config = read_config(args.config)
    for key in ("train_hr", "val_hr"):
        if getattr(args, key):
            config["data"][key] = getattr(args, key)
        if not config["data"][key]:
            parser.error(f"data.{key} is required")
        config["data"][key] = str(Path(config["data"][key]).resolve())
    if args.output:
        config["output"] = args.output
    config["output"] = str(Path(config["output"]).resolve())
    if args.device:
        config["runtime"]["device"] = args.device
    if config["pca"]["basis"]:
        config["pca"]["basis"] = str(Path(config["pca"]["basis"]).resolve())
    if args.smoke:
        config["data"].update(crop_size=config["data"]["scale"] * 8, repeat=1, batch_size=1, workers=0)
        config["analysis"].update(ranks=[1, config["data"]["scale"]**2 - 1], contexts=[1, 3],
                                   train_samples_per_crop=16, val_samples_per_image=16)
        config["cnn"].update(contexts=[3], width=4, epochs=2, seeds=[config["runtime"]["seed"]])
        config["runtime"]["max_train_batches"] = 2
        config["evaluation"].update(max_val_images=1, lpips=False)
        config["pca"]["repeat"] = 1
        config["benchmark"].update(lr_size=[8, 8], warmup=1, iterations=2)
        config["sweep"].update(ranks=[config["data"]["scale"]**2 - 1], seeds=[config["runtime"]["seed"]], epochs=1,
                                 variants=[{"name": "nuse_smoke", "backbone": "nuse", "width": 4, "num_blocks": 2}])
    runner = ResearchRunner(config)
    if args.stage != "report":
        runner.ensure_basis()
    if args.stage == "all":
        runner.oracle()
        runner.ridge()
        runner.cnn()
        runner.report()
    elif args.stage != "basis":
        getattr(runner, args.stage)()


if __name__ == "__main__":
    main()
