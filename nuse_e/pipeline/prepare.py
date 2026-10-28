"""Bước setup (dữ liệu, cache, checkpoint cũ) và stats (thống kê PCA, suy kiến trúc)."""
import shutil
import time
from pathlib import Path

import torch

from ..core.io import write_json
from ..nullspace.architecture import architecture_from_stats
from ..nullspace.statistics import fit_statistics

def step_setup(ctx):
    """Tìm dữ liệu, tạo cache, kiểm checkpoint cũ; chép stats/ và runs/ của phiên trước nếu có --resume-from."""
    c = ctx.c
    if ctx.args.resume_from:
        src = Path(ctx.args.resume_from).expanduser().resolve()
        for sub in ("stats", "runs"):
            if (src / sub).is_dir():
                shutil.copytree(src / sub, ctx.out / sub, dirs_exist_ok=True)
        print("Đã chép stats/ và runs/ từ", src, "(file đã có không bị xoá; run có DONE sẽ không train lại)")
    ctx.data()
    protocol = {"epochs": c.EPOCHS, "batch_size": c.BATCH, "repeat": c.REPEAT, "lr": c.LR, "min_lr": c.MIN_LR,
                "grad_clip": c.GRAD_CLIP}
    for label, path in ctx.old().items():
        cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
        method = ((cfg.get("train_dataset_options") or {}).get("resize") or {}).get("method", "average_pool")
        diff = {k: (cfg.get(k), v) for k, v in protocol.items() if cfg.get(k) != v}
        print(f"{label:28s} {path}\n   LR = {method}; khác giao thức: {diff or 'không'}")
    if not ctx.old():
        print("Không có checkpoint cũ (--old-runs): baseline dùng số đã báo cáo [số cũ], trần latency dùng số cũ.")
    print("Thiết bị:", ctx.device, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print("Output:", ctx.out)


def step_stats(ctx):
    """PCA null space làm đối xứng D4, Wiener, R² k-fold; rồi suy kiến trúc của mọi biến thể (không chọn tay)."""
    c = ctx.c
    for method in c.METHODS:
        folder = ctx.stats_dir / method
        stats_path = folder / "stats.pt"
        if not stats_path.exists():
            t0 = time.time()
            stats = fit_statistics(ctx.data()["train_cache"], method, ctx.device, repeat=c.STATS_REPEAT,
                                     folds=c.STATS_FOLDS, workers=c.WORKERS, seed=c.SEED)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(stats, stats_path)
            print(f"[{method}] thống kê xong sau {time.time() - t0:.0f}s")
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        lam = stats["eigenvalues"].mean(0)
        print(f"\n=== {method}: d = {stats['d']}, sai lệch bất biến D4 = {stats.get('d4_invariance', float('nan')):.1e}")
        print("tỉ số λ_i/λ_(i+1):", [round(x, 3) for x in (lam[:-1] / lam[1:]).tolist()])
        print("R² tuyến tính (trung bình kênh):", [round(x, 3) for x in stats["r2"].mean(0).tolist()])
        for name in c.NUSE:
            arch = architecture_from_stats(stats, **ctx.design_of(name))
            write_json(ctx.arch_path(method, name), arch)
            print(f"{name:14s} W={arch['width']} {arch['block']:5s} N={sum(arch['blocks'])} alpha={arch['alpha']:7s} | "
                  f"K = {arch['budgets']} | block = {arch['blocks']} | L_g = {arch['linear']} | "
                  f"năng lượng = {[round(e, 3) for e in arch['retained_energy']]}")
