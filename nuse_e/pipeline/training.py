"""Bước train: chạy mọi run còn thiếu, mỗi GPU một tiến trình; pha A rồi pha B."""
import os
import subprocess
import sys
import time

import torch

from ..core.io import write_json
from .context import PACKAGE_PARENT

def step_train(ctx):
    """Train mọi run còn thiếu: pha A (mọi biến thể, LR average pool), rồi pha B (bicubic cho biến thể tốt nhất)."""
    c, data = ctx.c, ctx.data()
    admissible = ctx.admissible(None)
    priority = {m: i for i, m in enumerate(c.MODELS)}
    no_train = set(c.BASELINES)

    def wanted(method, model):
        if model.startswith("NuSE-E"):
            if model not in admissible:
                return False
            return method == "average_pool" or model == "NuSE-E" or "average_pool" not in c.METHODS
        return model not in no_train and not ctx.uses_old(method, model)

    def best_psnr_kmax(method, model):
        ck = torch.load(ctx.run_dir(method, model) / "best.pt", map_location="cpu", weights_only=False)
        kmax = ck["arch"]["budgets"][-1]
        rec = next(r for r in ck["history"] if r["epoch"] == ck["epoch"])
        return rec["validation"][f"rgb_psnr_K{kmax}"]

    def job_config(method, model):
        nuse = model.startswith("NuSE-E")
        cfg = dict(model=model, method=method, run_dir=str(ctx.run_dir(method, model)),
                   train_files=[str(p) for p in data["train_files"]], val_files=[str(p) for p in data["val_files"]],
                   train_cache=str(data["train_cache"]),
                   stats=str(ctx.stats_dir / method / "stats.pt") if nuse else None,
                   arch=str(ctx.arch_path(method, model)) if nuse else None,
                   epochs=c.EPOCHS, batch=c.BATCH, repeat=c.REPEAT, lr=c.LR, min_lr=c.MIN_LR, grad_clip=c.GRAD_CLIP,
                   val_every=c.VAL_EVERY, checkpoint_every=c.CHECKPOINT_EVERY, workers=c.WORKERS, seed=c.SEED,
                   amp=c.AMP, quick_val_images=c.QUICK_VAL_IMAGES, quick_val_crop=c.QUICK_VAL_CROP,
                   device="cuda" if ctx.device.type == "cuda" else "cpu", max_iters=c.MAX_ITERS)
        path = ctx.run_dir(method, model) / "config.json"
        write_json(path, cfg)
        return path

    jobs = sorted([(m, mdl) for m in c.METHODS for mdl in c.MODELS if wanted(m, mdl) and not ctx.is_done(m, mdl)],
                  key=lambda j: (priority[j[1]], j[0] != "average_pool"))
    phase_b = "bicubic" in c.METHODS and "average_pool" in c.METHODS and c.PHASE_B_TOP > 0
    print("Đã xong từ trước (DONE):", [f"{mdl} / {m}" for m in c.METHODS for mdl in c.MODELS if ctx.is_done(m, mdl)] or "không có")
    print("Cần train:", jobs or "không còn run nào")
    if ctx.args.gpus:
        slots = ctx.args.gpus.split(",")
    else:
        slots = [str(i) for i in range(torch.cuda.device_count())] if ctx.device.type == "cuda" else ["cpu"]
    start, active, shown = time.time(), {}, {}
    while jobs or active or phase_b:
        if not jobs and not active and phase_b:
            phase_b = False
            done_a = [n for n in c.NUSE if n != "NuSE-E" and n in admissible and ctx.is_done("average_pool", n)]
            ranked = sorted(done_a, key=lambda n: -best_psnr_kmax("average_pool", n))
            jobs = [("bicubic", n) for n in ranked[:c.PHASE_B_TOP] if not ctx.is_done("bicubic", n)]
            print("Pha B (LR bicubic) cho biến thể tốt nhất pha A:", jobs or "không có")
        for slot in [s for s in slots if s not in active]:
            if not jobs:
                break
            if c.MAX_HOURS and (time.time() - start) / 3600 > c.MAX_HOURS:
                print(f"Quá {c.MAX_HOURS} giờ: không nhận run mới. Chạy lại bước train để tiếp tục.")
                jobs, phase_b = [], False
                break
            method, model = jobs.pop(0)
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(PACKAGE_PARENT), os.environ.get("PYTHONPATH")])))
            if slot != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = slot
            config_path = job_config(method, model)
            log = open(ctx.run_dir(method, model) / "stdout.log", "a")
            active[slot] = (method, model, subprocess.Popen(
                [sys.executable, "-u", "-m", "nuse_e.training.run", str(config_path)],
                env=env, stdout=log, stderr=subprocess.STDOUT), log)
            print(f"[{slot}] bắt đầu {model} / {method}")
        time.sleep(1 if c.SMOKE else 30)
        for slot, (method, model, proc, log) in list(active.items()):
            tail = ctx.run_dir(method, model) / "train.log"
            text = tail.read_text().strip() if tail.exists() else ""
            last = text.splitlines()[-1] if text else ""
            if last and shown.get(slot) != last:
                print(f"[{slot}] {model} / {method}: {last}", flush=True)
                shown[slot] = last
            if proc.poll() is not None:
                log.close()
                del active[slot]
                if proc.returncode != 0:
                    print((ctx.run_dir(method, model) / "stdout.log").read_text()[-3000:])
                    sys.exit(f"{model} / {method} lỗi (mã {proc.returncode})")
    print(f"Train xong sau {(time.time() - start) / 3600:.2f} giờ")
