"""Bước curves (đường học) và figures (hình so sánh)."""
import json
import statistics

import torch

from ..data.images import ValImages, as_float
from ..evaluation.metrics import psnr
from ..operators import degrade
from .context import name_of
from .report import Tables

def plt_module():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def step_curves(ctx):
    """Hình 1: PSNR DIV2K valid (đủ 100 ảnh) theo epoch.

    Nét liền: bản v2 ở mọi exit K (một model). Nét chấm: mỗi biến thể ở exit lớn nhất (mỗi đường một model).
    """
    plt, c = plt_module(), ctx.c
    fig, axes = plt.subplots(1, len(c.METHODS), figsize=(7 * len(c.METHODS), 4.2), squeeze=False)
    for ax, method in zip(axes[0], c.METHODS):
        for model in c.MODELS:
            path = ctx.run_dir(method, model) / "history.json"
            if not path.exists():
                continue
            hist = [r for r in json.loads(path.read_text()) if "validation" in r]
            if model == "NuSE-E":
                for k in ctx.arch(method, model)["budgets"]:
                    ax.plot([r["epoch"] for r in hist], [r["validation"][f"rgb_psnr_K{k}"] for r in hist],
                            label=f"v2 K={k}", lw=1.6)
            elif model.startswith("NuSE-E"):
                k = ctx.arch(method, model)["budgets"][-1]
                ax.plot([r["epoch"] for r in hist], [r["validation"][f"rgb_psnr_K{k}"] for r in hist],
                        ":", label=f"{model.split('/')[1]} K={k}", lw=1.4)
            else:
                ax.plot([r["epoch"] for r in hist], [r["validation"]["rgb_psnr"] for r in hist], "--", label=model, lw=1.4)
        ax.set(title=f"LR {method}", xlabel="epoch", ylabel="PSNR DIV2K valid (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="liền: v2 mọi K | chấm: biến thể, K lớn nhất", title_fontsize=7)
    plt.tight_layout()
    fig.savefig(ctx.fig / "1_training_curves.png", dpi=150)
    plt.close(fig)
    print("->", ctx.fig / "1_training_curves.png")


def step_figures(ctx):
    """Hình 2: PSNR theo chi phí; 3: hai loại LR; 4: phân rã latency; 5: từng ảnh; 6: crop trực quan."""
    import numpy as np
    from torch.nn import functional as F
    plt, c, t = plt_module(), ctx.c, Tables(ctx)
    colors = {"SPANV2": "#2a78d6", "DISP": "#eb6834", "PLKSR-Tiny": "#1baf7a", "Bicubic": "#8a8984"}
    palette = ["#e34948", "#b3267a", "#8f4fc9", "#d98a1a", "#5c6b73", "#2d9a8f", "#a0522d", "#7a7a1a", "#1f4e8c", "#c71585"]
    for method in c.METHODS:
        sel = t.selected.get(method)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, xkey, xlabel in [(axes[0], "lat", f"latency fp16 @LR {t.ref} (ms)"),
                                 (axes[1], "gflops", f"GFLOPs @LR {c.FLOPS_SIZE[0]}²")]:
            nuse = {}
            for (s, lab), r in t.report[method].items():
                if s != "DIV2K-valid" or lab == "Bicubic":
                    continue
                x = t.lat(method, lab, t.ref, "fp16") if xkey == "lat" else (t.cost(method, lab) or {}).get("gflops")
                if x is None:
                    continue
                if lab.startswith("NuSE-E"):
                    nuse.setdefault(name_of(lab), []).append((x, r["rgb_psnr"], lab.split("=")[1]))
                else:
                    base = lab.split(" +")[0]
                    marker = "s" if "[số cũ]" in lab else "^" if "(cũ)" in lab or "NuSE k=8" in lab else "o"
                    ax.scatter(x, r["rgb_psnr"], s=60, color=colors.get(base.replace(" [số cũ]", ""), "#6250d6"), marker=marker)
                    ax.annotate(lab, (x, r["rgb_psnr"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
            for i, (name, pts) in enumerate(nuse.items()):
                pts.sort()
                main = name in ("NuSE-E", sel)
                color = palette[i % len(palette)]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o" if main else ":o", color=color,
                        lw=1.8 if main else 1.0, ms=6 if main else 4, label=name + (" ★" if name == sel else ""))
                if main:
                    for x, y, k in pts:
                        ax.annotate(f"K={k}", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=8, color=color)
            ax.set(xlabel=xlabel, ylabel="PSNR DIV2K valid (dB)", title=f"LR {method}")
            if xkey == "gflops":
                ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(ctx.fig / f"2_psnr_vs_cost_{method}.png", dpi=150)
        plt.close(fig)

    if any(not r["device"].startswith("cpu") for r in t.latency):
        fig, axes = plt.subplots(1, len(c.METHODS), figsize=(7 * len(c.METHODS), 4.5), squeeze=False)
        for ax, method in zip(axes[0], c.METHODS):
            full = ctx.full(method)
            labels = [lab for (s, lab) in t.report[method] if s == "DIV2K-valid" and lab != "Bicubic"
                      and "(cũ)" not in lab and "[số cũ]" not in lab and name_of(lab) in full]
            pos = np.arange(len(labels))
            pred = [t.lat(method, lab, t.ref, "fp16", "predictor") or t.lat(method, lab, t.ref, "fp16") or 0 for lab in labels]
            dec = [t.lat(method, lab, t.ref, "fp16", "decoder + consistency") or 0 for lab in labels]
            ax.bar(pos, pred, 0.6, label="mạng / predictor")
            ax.bar(pos, dec, 0.6, bottom=pred, label="decoder + consistency (NuSE-E)")
            ax.set_xticks(pos, labels, rotation=30, ha="right", fontsize=8)
            ax.set(ylabel="ms (median, fp16)", title=f"Latency @LR {t.ref}, LR {method}")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(ctx.fig / "4_latency_breakdown.png", dpi=150)
        plt.close(fig)

    cross = t.cross()
    labels = [k for k in cross if "(cũ)" not in k]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (key, title) in zip(axes, [("rgb_psnr", "PSNR ↑"), ("rgb_ssim", "SSIM ↑"), ("lpips", "LPIPS ↓"), ("fid_patch256", "FID ↓")]):
        width = 0.8 / len(c.METHODS)
        for i, m in enumerate(c.METHODS):
            vals = [cross[k][m][1].get(key, np.nan) if m in cross[k] else np.nan for k in labels]
            ax.bar(np.arange(len(labels)) + i * width, vals, width, label=f"LR {m}")
        ax.set_xticks(np.arange(len(labels)) + width * (len(c.METHODS) - 1) / 2, labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if key in ("rgb_psnr", "rgb_ssim"):
            finite = [cross[k][m][1].get(key) for m in c.METHODS for k in labels if m in cross[k] and cross[k][m][1].get(key) is not None]
            if finite:
                ax.set_ylim(min(finite) - (0.5 if key == "rgb_psnr" else 0.02), max(finite) + (0.3 if key == "rgb_psnr" else 0.01))
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(ctx.fig / "3_avgpool_vs_bicubic.png", dpi=150)
    plt.close(fig)

    crop = 192
    val_files = ctx.data()["val_files"]
    for method in c.METHODS:
        sel = t.selected.get(method)
        per_image = ctx.per_image(method)
        if not sel:
            continue
        ours_label = f"{sel} K={ctx.arch(method, sel)['budgets'][-1]}"
        ours = per_image.get(("DIV2K-valid", ours_label))
        others = [lab for (s, lab) in per_image if s == "DIV2K-valid" and not lab.startswith("NuSE-E")]
        v2_label = f"NuSE-E K={ctx.arch(method, 'NuSE-E')['budgets'][-1]}"
        if sel != "NuSE-E" and ("DIV2K-valid", v2_label) in per_image:
            others.append(v2_label)
        if not ours or not others:
            continue
        fig, axes = plt.subplots(1, len(others), figsize=(5.5 * len(others), 4), squeeze=False)
        for ax, base in zip(axes[0], others):
            theirs = {r["name"]: r for r in per_image[("DIV2K-valid", base)]}
            gain = [r["rgb_psnr"] - theirs[r["name"]]["rgb_psnr"] for r in ours]
            ax.hist(gain, bins=25, color="#2a78d6")
            ax.axvline(0, color="k", lw=1)
            ax.set(title=f"{ours_label} − {base}: thắng {sum(g > 0 for g in gain)}/{len(gain)}, "
                         f"trung vị {statistics.median(gain):+.3f} dB", xlabel="ΔPSNR (dB)")
            ax.title.set_fontsize(9)
        plt.tight_layout()
        fig.savefig(ctx.fig / f"5_per_image_{method}.png", dpi=150)
        plt.close(fig)

        models = ctx.models_eval(method)
        base = others[0]
        theirs = {r["name"]: r for r in per_image[("DIV2K-valid", base)]}
        ranked = sorted(ours, key=lambda r: r["rgb_psnr"] - theirs[r["name"]]["rgb_psnr"])
        stems = [p.stem for p in val_files]
        for pick in (ranked[-1]["name"], ranked[0]["name"]):
            hr = as_float(ValImages([val_files[stems.index(pick)]])[0][0], ctx.device)
            lr = degrade(hr, method)
            with torch.no_grad():
                outs = {"Bicubic": F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)}
                for label, (model, k) in models.items():
                    if label not in (ours_label, *others):
                        continue
                    outs[label] = model(lr, k) if k else model(lr)
            diff = F.avg_pool2d((outs[ours_label] - outs[base]).abs().mean(1, keepdim=True), min(crop, hr.shape[-2]), stride=crop // 4)
            row, col = divmod(int(diff.flatten().argmax()), diff.shape[-1])
            top, left = row * (crop // 4), col * (crop // 4)
            fig, axes = plt.subplots(1, len(outs) + 1, figsize=(3.6 * (len(outs) + 1), 4))
            for ax, (title, img) in zip(axes, [*outs.items(), ("HR", hr)]):
                ax.imshow(img[0, :, top:top + crop, left:left + crop].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
                ax.axis("off")
                ax.set_title(title if title == "HR" else f"{title}\n{psnr(img, hr).item():.2f} dB", fontsize=9)
            fig.suptitle(f"LR {method} | {pick}")
            plt.tight_layout()
            fig.savefig(ctx.fig / f"6_visual_{method}_{pick}.png", dpi=150)
            plt.close(fig)
    print("Hình ->", ctx.fig)
