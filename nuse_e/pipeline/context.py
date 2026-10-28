"""Bối cảnh dùng chung của mọi bước: đường dẫn, thiết lập, trạng thái đã ghi trên đĩa."""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from .. import config as CONFIG
from ..core.constants import IMAGE_EXT
from ..data.cache import build_flat_cache
from ..data.images import image_files, to_bmp
from ..models.registry import budgets_of, load_run

PACKAGE_PARENT = Path(__file__).resolve().parents[2]   # thư mục chứa package nuse_e (gốc repo)
PATH_KEYS = ("data_root", "train_hr", "val_hr", "old_runs", "benchmarks", "cache")


class Ctx:
    """Đường dẫn, thiết lập và trạng thái đã ghi trên đĩa của một thư mục output."""

    def __init__(self, args):
        self.args = args
        self.c = CONFIG.load(args.smoke, args.variants.split(",") if args.variants else None)
        self.out = Path(args.out).resolve()
        self.stats_dir, self.runs, self.compare = self.out / "stats", self.out / "runs", self.out / "comparison"
        self.fig = self.compare / "figures"
        for path in (self.out, self.stats_dir, self.runs, self.compare, self.fig):
            path.mkdir(parents=True, exist_ok=True)
        self.paths = self._paths(args)
        self.cache = Path(self.paths.get("cache") or self.out / "cache")
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._data, self._old = None, None

    # ---- đường dẫn dữ liệu: nhớ trong <out>/paths.json để các bước sau khỏi truyền lại
    def _paths(self, args):
        file = self.out / "paths.json"
        paths = json.loads(file.read_text()) if file.exists() else {}
        for key in PATH_KEYS:
            value = getattr(args, key, None)
            if value:
                paths[key] = str(Path(value).expanduser().resolve())
        file.write_text(json.dumps(paths, indent=2, ensure_ascii=False))
        return paths

    def find_dir(self, words, minimum, exclude=("_lr", "lr_", "x2", "x3", "x4", "x8"), root=None):
        root = Path(root or self.paths.get("data_root") or "")
        if not root or not root.is_dir():
            return None
        best = None
        for folder_root, dirs, _ in os.walk(root, followlinks=True):
            for name in dirs:
                folder = Path(folder_root) / name
                text = str(folder).lower()
                if any(w in text for w in exclude) or not all(w in text for w in words):
                    continue
                n = image_count(folder)
                if n >= minimum and (best is None or n > best[1]):
                    best = (folder, n)
        return best[0] if best else None

    # ---- dữ liệu: cache uint8 phẳng cho train, BMP cho valid (tạo một lần, dùng lại)
    def data(self):
        if self._data:
            return self._data
        c = self.c
        train_src = self.paths.get("train_hr") or self.find_dir(("div2k", "train", "hr"), 6 if c.SMOKE else 500)
        val_src = self.paths.get("val_hr") or self.find_dir(("div2k", "valid", "hr"), 2 if c.SMOKE else 50)
        if not train_src or not val_src:
            sys.exit("Không tìm thấy DIV2K train/valid HR. Chạy: python -m nuse_e.pipeline setup "
                     "--train-hr <DIV2K_train_HR> --val-hr <DIV2K_valid_HR> (hoặc --data-root <thư mục chứa DIV2K>)")
        t0 = time.time()
        workers = max(1, os.cpu_count() or 1)
        train_files = image_files(train_src)
        train_cache = build_flat_cache(train_files, self.cache / "div2k_train_u8.bin", workers=workers)
        folder = self.cache / "div2k_valid_bmp"
        folder.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            val_files = [Path(p) for p in pool.map(to_bmp, [(str(p), str(folder)) for p in image_files(val_src)],
                                                   chunksize=8)]
        benchmarks = {}
        bench_root = self.paths.get("benchmarks") or self.paths.get("data_root")
        for name in c.BENCHMARK_NAMES:
            found = self.find_dir((name.lower(),), 5, root=bench_root,
                                  exclude=("_lr", "lr_", "lrbi", "x2", "x3", "x4", "x8", "bicubic"))
            if found:
                benchmarks[name] = image_files(found)
        print(f"Train HR: {train_src} ({len(train_files)} ảnh, cache {train_cache.stat().st_size / 1e9:.2f} GB) | "
              f"Valid HR: {val_src} ({len(val_files)} ảnh) | benchmark: {({k: len(v) for k, v in benchmarks.items()} or 'không có')} "
              f"| {time.time() - t0:.0f}s")
        self._data = dict(train_files=train_files, train_cache=train_cache, val_files=val_files, benchmarks=benchmarks)
        return self._data

    # ---- checkpoint của thí nghiệm cũ (baseline, bản cắm NuSE k=8)
    def old(self):
        if self._old is None:
            root = Path(self.paths["old_runs"]) if self.paths.get("old_runs") else None
            self._old = {}
            for label, folder in self.c.OLD_NAMES.items():
                hits = sorted(root.rglob(f"{folder}/best.pt")) if root and root.is_dir() else []
                if hits:
                    self._old[label] = hits[0]
        return self._old

    # ---- tên, thư mục, kiến trúc
    def design_of(self, model):
        return {**self.c.DESIGN, **self.c.VARIANTS[model.split("/")[1]]} if "/" in model else dict(self.c.DESIGN)

    def run_dir(self, method, model):
        return self.runs / method / slug(model)

    def arch_path(self, method, model):
        return self.stats_dir / method / ("arch.json" if model == "NuSE-E" else f"arch_{model.split('/')[1]}.json")

    def arch(self, method, model):
        path = self.arch_path(method, model)
        if not path.exists():
            sys.exit(f"Chưa có {path}: chạy bước stats trước.")
        return json.loads(path.read_text())

    def is_done(self, method, model):
        return (self.run_dir(method, model) / "DONE").exists()

    def uses_old(self, method, model):
        return self.c.OLD_RUNS == "reuse" and method == "average_pool" and model in self.old()

    # ---- trạng thái ghi bởi các bước trước
    def screen(self):
        path = self.compare / "screen.json"
        if not path.exists():
            sys.exit("Chưa có comparison/screen.json: chạy bước screen trước.")
        return json.loads(path.read_text())

    def admissible(self, method):
        return [r["model"] for r in self.screen()["rows"] if r["admissible"]]

    def selected(self):
        path = self.compare / "selected.json"
        if not path.exists():
            sys.exit("Chưa có comparison/selected.json: chạy bước select trước.")
        return json.loads(path.read_text())

    def full(self, method):
        """Model được đánh giá đầy đủ (LPIPS, FID, latency mọi K): baseline, bản v2, bản được chọn."""
        return {*self.c.BASELINES, "NuSE-E", self.selected().get(method)} - {None}

    def report(self, method):
        path = self.compare / method / "report.json"
        if not path.exists():
            sys.exit(f"Chưa có {path}: chạy bước evaluate trước.")
        return {(r["set"], r["model"]): {k: v for k, v in r.items() if k not in ("set", "model")}
                for r in json.loads(path.read_text())}

    def per_image(self, method):
        path = self.compare / method / "per_image.json"
        rows = json.loads(path.read_text()) if path.exists() else []
        out = {}
        for r in rows:
            out.setdefault((r["set"], r["model"]), []).append({k: v for k, v in r.items() if k not in ("set", "model")})
        return out

    def load_list(self, name):
        path = self.compare / name
        return json.loads(path.read_text()) if path.exists() else []

    def models_eval(self, method):
        """{nhãn: (model, k)} của mọi model có checkpoint cho loại LR này."""
        entries = {}
        for model_name in self.c.MODELS:
            path = self.old()[model_name] if self.uses_old(method, model_name) else self.run_dir(method, model_name) / "best.pt"
            if not Path(path).exists() or (model_name.startswith("NuSE-E") and not self.is_done(method, model_name)):
                print(f"không có {model_name} / {method} -> bỏ qua")
                continue
            model, _ = load_run(path)
            model = model.to(self.device).eval()
            for k in budgets_of(model):
                entries[model_name if k is None else f"{model_name} K={k}"] = (model, k)
        if method == "average_pool" and self.c.SHOW_OLD_PLUGIN:
            for label, path in self.old().items():
                if "NuSE k=8" in label:
                    entries[label] = (load_run(path)[0].to(self.device).eval(), None)
        return entries


def slug(model):
    return {"SPANV2": "spanv2", "DISP": "disp", "PLKSR-Tiny": "plksr_tiny"}.get(model, "nuse_e" + model[6:].replace("/", "_"))


def name_of(label):
    return label.split(" K=")[0]


def image_count(folder):
    try:
        return sum(1 for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)
    except OSError:
        return 0


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def fmt(v, d=4):
    return "—" if v is None or v != v else ("∞" if v == float("inf") else f"{v:.{d}f}")
