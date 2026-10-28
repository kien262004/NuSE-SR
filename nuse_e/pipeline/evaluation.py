"""Bước evaluate (chất lượng ảnh) và cost (tham số, GFLOPs, latency)."""
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from ..core.io import write_csv, write_json
from ..data.images import ValImages
from ..evaluation.cost import count_params, gflops, stage_functions, time_cpu, time_cuda
from ..evaluation.evaluate import evaluate, evaluate_bicubic_upsampling
from ..evaluation.perceptual import LPIPSMetric, PatchFID
from .context import name_of

def metric_objects(ctx):
    lpips = fid_cls = None
    if ctx.c.USE_LPIPS:
        try:
            lpips = LPIPSMetric().to(ctx.device)
        except ImportError:
            print("Thiếu gói lpips: bỏ LPIPS (pip install lpips)")
    if ctx.c.USE_FID:
        try:
            PatchFID(ctx.device)
            fid_cls = PatchFID
        except (ImportError, ModuleNotFoundError) as error:
            print(f"Thiếu gói cho FID ({error}): bỏ FID (pip install torchmetrics torch-fidelity)")
    return lpips, fid_cls


def step_evaluate(ctx):
    """PSNR/SSIM RGB (SR thô, shave 0), kênh Y (uint8, shave 4), LPIPS, FID patch 256, max|A(SR) - y|."""
    c, data = ctx.c, ctx.data()
    full = {m: ctx.full(m) for m in c.METHODS}
    lpips, fid_cls = metric_objects(ctx)
    val_images = [img for img in ValImages(data["val_files"])]
    for method in c.METHODS:
        models = ctx.models_eval(method)
        report, per_image = {}, {}
        sets = {"DIV2K-valid": val_images, **{n: [img for img in ValImages(f)] for n, f in data["benchmarks"].items()}}
        for set_name, images in sets.items():
            fid = fid_cls(ctx.device) if fid_cls and len(images) >= 50 else None
            summary, _ = evaluate_bicubic_upsampling(images, method, ctx.device, lpips=lpips, fid=fid, fid_real=True)
            report[(set_name, "Bicubic")] = summary
            for label, (model, k) in models.items():
                t0 = time.time()
                is_full = name_of(label) in full[method] or "(cũ)" in label
                save = (ctx.out / "sr_images" / method / label.replace(" ", "_")
                        if c.SAVE_SR_IMAGES and set_name == "DIV2K-valid" else None)
                summary, rows = evaluate(model, images, method, ctx.device, k=k, lpips=lpips if is_full else None,
                                           fid=fid if is_full else None, save_dir=save)
                report[(set_name, label)] = summary
                per_image[(set_name, label)] = rows
                print(f"[{method}] {set_name:12s} {label:30s} PSNR {summary['rgb_psnr']:.4f}  SSIM {summary['rgb_ssim']:.4f}  "
                      f"LPIPS {summary.get('lpips', float('nan')):.4f}  FID {summary.get('fid_patch256', float('nan')):.2f}  "
                      f"({time.time() - t0:.0f}s)")
        if method == "average_pool":
            report.update(reported_baselines(ctx, models))
        folder = ctx.compare / method
        rows = [{"set": s, "model": lab, **v} for (s, lab), v in report.items()]
        write_csv(folder / "quality.csv", rows)
        write_json(folder / "report.json", rows)
        flat = [{"set": s, "model": lab, **r} for (s, lab), rr in per_image.items() for r in rr]
        write_csv(folder / "per_image.csv", flat)
        write_json(folder / "per_image.json", flat)
        if method == "average_pool":
            check_old_checkpoints(ctx, report)
        quality_by_config(ctx, method, models, val_images, full[method])


def reported_baselines(ctx, models):
    """Baseline không có checkpoint: dùng số đã báo cáo, nhãn [số cũ]."""
    label_of = {"SPANV2": "SPANV2", "DISP": "DISP", "PLKSR-Tiny": "PLKSR-Tiny", "SPANV2 + NuSE": "SPANV2 + NuSE k=8 (cũ)",
                "DISP + NuSE": "DISP + NuSE k=8 (cũ)", "PLKSR-Tiny + NuSE": "PLKSR-Tiny + NuSE k=8 (cũ)"}
    out, rows = {}, []
    for key, r in ctx.c.OLD_REPORTED.items():
        label = label_of[key]
        if label in models or ("(cũ)" in label and not ctx.c.SHOW_OLD_PLUGIN):
            continue
        name = label.replace(" (cũ)", "") + " [số cũ]"
        out[("DIV2K-valid", name)] = {k: r[k] for k in ("rgb_psnr", "rgb_ssim", "lpips", "lr_max_abs")}
        rows.append({"model": name, **r})
        print(f"{name:30s} PSNR {r['rgb_psnr']:.4f}  SSIM {r['rgb_ssim']:.4f} (số đã báo cáo; latency cũ trên {r['gpu']})")
    write_json(ctx.compare / "reported_rows.json", rows)
    return out


def check_old_checkpoints(ctx, report):
    """PSNR đo lại của checkpoint cũ phải khớp summary.json cũ (dưới 0.01 dB)."""
    keys = {"SPANV2": "SPANV2", "DISP": "DISP", "PLKSR-Tiny": "PLKSR-Tiny", "SPANV2 + NuSE k=8 (cũ)": "SPANV2 + NuSE",
            "DISP + NuSE k=8 (cũ)": "DISP + NuSE", "PLKSR-Tiny + NuSE k=8 (cũ)": "PLKSR-Tiny + NuSE"}
    for label, path in ctx.old().items():
        summary_path = Path(path).parent.parent / "comparison" / "summary.json"
        mine = report.get(("DIV2K-valid", label))
        if not summary_path.exists() or mine is None:
            continue
        old = json.loads(summary_path.read_text())["reports"].get(keys[label])
        if old:
            delta = mine["rgb_psnr"] - old["rgb_psnr"]
            print(f"{'OK ' if abs(delta) < 0.01 else 'LỆCH'} {label:28s} cũ {old['rgb_psnr']:.4f} | "
                  f"đo lại {mine['rgb_psnr']:.4f} | {delta:+.4f} dB")


def quality_by_config(ctx, method, models, val_images, full):
    """PSNR theo độ chính xác số (fp32/fp16) và kiểu output (float/uint8), DIV2K valid."""
    path = ctx.compare / "quality_by_config.json"
    rows = [r for r in (json.loads(path.read_text()) if path.exists() else []) if r["lr"] != method]
    for label, (model, k) in models.items():
        if "(cũ)" in label or name_of(label) not in full:
            continue
        for amp in ([False, True] if ctx.device.type == "cuda" else [False]):
            s, _ = evaluate(model, val_images, method, ctx.device, k=k, amp=amp)
            p = "fp16" if amp else "fp32"
            rows += [{"lr": method, "model": label, "precision": p, "output": "float", "psnr": s["rgb_psnr"], "ssim": s["rgb_ssim"]},
                     {"lr": method, "model": label, "precision": p, "output": "uint8", "psnr": s["rgb_psnr_uint8"], "ssim": float("nan")}]
    write_csv(ctx.compare / "quality_by_config.csv", rows)
    write_json(path, rows)


def step_cost(ctx):
    """Tham số học được, GFLOPs @LR 256², latency GPU (batch 1, CUDA events, xen kẽ nhiều vòng) và CPU."""
    c = ctx.c
    torch.backends.cudnn.benchmark = True
    cost, latency = [], []
    for method in c.METHODS:
        models, full = ctx.models_eval(method), ctx.full(method)
        for label, (model, k) in models.items():
            cost.append({"lr": method, "model": label, "params": count_params(model, k),
                         "gflops": gflops(model, c.FLOPS_SIZE, ctx.device, k)})
        if ctx.device.type == "cuda":
            for size in c.LATENCY_SIZES:
                x = torch.rand(1, 3, *size, device=ctx.device)
                fns = {}
                for label, (model, k) in models.items():
                    if "(cũ)" in label or (k is not None and k != model.budgets[0]):
                        continue
                    with torch.no_grad():
                        entries = stage_functions(name_of(label), model, x)
                    if k is not None and name_of(label) not in full:      # biến thể không chọn: chỉ K lớn nhất
                        entries = {key: fn for key, fn in entries.items() if key[0].endswith(f" K={model.budgets[-1]}")}
                    fns.update(entries)
                for precision in c.LATENCY_PRECISIONS:
                    times, memory = time_cuda(fns, precision, c.WARMUP, c.ITERS, c.ROUNDS)
                    for (label, stage), values in times.items():
                        latency.append({"lr": method, "device": torch.cuda.get_device_name(0), "lr_size": f"{size[0]}x{size[1]}",
                                        "precision": precision, "model": label, "stage": stage,
                                        "median_ms": statistics.median(values), "p90_ms": sorted(values)[int(0.9 * len(values))],
                                        "peak_mem_mb": memory[(label, stage)]})
        if c.CPU_LATENCY_SIZE:
            x = torch.rand(1, 3, *c.CPU_LATENCY_SIZE)
            for label, (model, k) in models.items():
                if "(cũ)" in label or name_of(label) not in full:
                    continue
                cpu_model = copy.deepcopy(model).cpu()
                with torch.no_grad():
                    values = time_cpu(lambda: cpu_model(x, k) if k else cpu_model(x), c.CPU_ITERS)
                latency.append({"lr": method, "device": f"cpu ({torch.get_num_threads()} threads)",
                                "lr_size": f"{c.CPU_LATENCY_SIZE[0]}x{c.CPU_LATENCY_SIZE[1]}", "precision": "fp32",
                                "model": label, "stage": "total", "median_ms": statistics.median(values),
                                "p90_ms": sorted(values)[int(0.9 * len(values))], "peak_mem_mb": float("nan")})
    for r in ctx.load_list("reported_rows.json"):
        cost.append({"lr": "average_pool", "model": r["model"], "params": r["params"], "gflops": r["gflops"]})
    write_csv(ctx.compare / "cost.csv", cost)
    write_json(ctx.compare / "cost.json", cost)
    if latency:
        write_csv(ctx.compare / "latency.csv", latency)
    write_json(ctx.compare / "latency.json", latency)
    print(len(cost), "dòng chi phí,", len(latency), "dòng latency"
          + ("" if ctx.device.type == "cuda" else " (không có GPU: chỉ đo CPU)"))
