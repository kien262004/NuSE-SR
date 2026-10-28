"""Bước report: bảng tổng hợp, ràng buộc cứng sau train, report.md."""
from ..core.io import write_csv, write_json
from .context import fmt, md_table, name_of

class Tables:
    """Truy vấn report / cost / latency đã ghi trên đĩa."""

    def __init__(self, ctx):
        self.ctx, c = ctx, ctx.c
        self.report = {m: ctx.report(m) for m in c.METHODS}
        self.cost_rows, self.latency = ctx.load_list("cost.json"), ctx.load_list("latency.json")
        self.selected = ctx.selected()
        self.ref = f"{c.LATENCY_SIZES[-1][0]}x{c.LATENCY_SIZES[-1][1]}"

    def lat(self, method, label, size, precision, stage="total"):
        for r in self.latency:
            if (r["lr"], r["model"], r["lr_size"], r["precision"], r["stage"]) == (method, label, size, precision, stage) \
                    and not r["device"].startswith("cpu"):
                return r["median_ms"]
        return None

    def cost(self, method, label):
        return next((r for r in self.cost_rows if r["lr"] == method and r["model"] == label), None)

    def cross(self):
        """NuSE-E (bản v2 và bản được chọn) ở K nhỏ nhất và K lớn nhất, cùng baseline, cho cả hai loại LR."""
        out = {}
        for m in self.ctx.c.METHODS:
            for (s, lab), r in self.report[m].items():
                key = lab
                if lab.startswith("NuSE-E"):
                    name = name_of(lab)
                    if name not in ("NuSE-E", self.selected.get(m)):
                        continue
                    k, budgets = int(lab.split("=")[1]), self.ctx.arch(m, name)["budgets"]
                    key = f"{name} (K lớn nhất)" if k == budgets[-1] else f"{name} (K nhỏ nhất)" if k == budgets[0] else None
                if s == "DIV2K-valid" and key:
                    out.setdefault(key, {})[m] = (lab, r)
        return out


def step_report(ctx):
    """Bảng tổng hợp, ràng buộc cứng sau train, so hai loại LR, phân rã latency -> comparison/report.md."""
    c, t = ctx.c, Tables(ctx)
    cap_ms = ctx.screen()["cap_ms"]
    flops = f"{c.FLOPS_SIZE[0]}x{c.FLOPS_SIZE[1]}"
    parts = []
    for method in c.METHODS:
        rows = []
        for (set_name, label), r in t.report[method].items():
            if set_name != "DIV2K-valid":
                continue
            co = t.cost(method, label)
            rows.append([label, fmt(r["rgb_psnr"]), fmt(r["rgb_ssim"]), fmt(r.get("y_psnr_shave4")), fmt(r.get("y_ssim_shave4")),
                         fmt(r.get("lpips")), fmt(r.get("fid_patch256"), 2), f"{r['lr_max_abs']:.1e}",
                         f"{co['params'] / 1e3:.1f}K" if co else "—", fmt(co["gflops"], 2) if co else "—",
                         fmt(t.lat(method, label, t.ref, "fp32"), 2), fmt(t.lat(method, label, t.ref, "fp16"), 2)])
        parts += [f"## LR = {method}, DIV2K valid", md_table(
            ["model", "PSNR ↑", "SSIM ↑", "Y-PSNR ↑", "Y-SSIM ↑", "LPIPS ↓", "FID ↓", "max|A(SR)−y|", "params",
             f"GFLOPs@{c.FLOPS_SIZE[0]}²", f"ms fp32 @{t.ref}", f"ms fp16 @{t.ref}"], rows)]
        bench = [(s, lab, r) for (s, lab), r in t.report[method].items() if s != "DIV2K-valid"]
        if bench:
            sets = sorted({s for s, _, _ in bench})
            labels = list(dict.fromkeys(lab for _, lab, _ in bench))
            parts += [f"### Benchmark (Y-PSNR / Y-SSIM, shave 4), LR = {method}", md_table(
                ["model", *sets], [[lab, *[next((f"{r['y_psnr_shave4']:.2f} / {r['y_ssim_shave4']:.4f}"
                                                 for s2, l2, r in bench if s2 == s and l2 == lab), "—") for s in sets]]
                                   for lab in labels])]
        write_json(ctx.compare / method / "summary.json",
                   {"reports": {f"{s} | {lab}": r for (s, lab), r in t.report[method].items()},
                    "cost": [r for r in t.cost_rows if r["lr"] == method],
                    "architecture": {n: ctx.arch(method, n) for n in c.NUSE if ctx.arch_path(method, n).exists()},
                    "selected": t.selected.get(method)})

    constraint_rows = []
    for method in c.METHODS:
        for name in c.NUSE:
            if not any(s == "DIV2K-valid" and name_of(lab) == name for (s, lab) in t.report[method]):
                continue
            arch = ctx.arch(method, name)
            ks = arch["budgets"]
            psnr = [t.report[method][("DIV2K-valid", f"{name} K={k}")]["rgb_psnr"] for k in ks]
            params = [(t.cost(method, f"{name} K={k}") or {}).get("params") for k in ks]
            ms = [t.lat(method, f"{name} K={k}", flops, "fp32") for k in ks]
            ms_known = [v for v in ms if v is not None]
            consistency = max(t.report[method][("DIV2K-valid", f"{name} K={k}")]["lr_max_abs"] for k in ks)
            checks = {"params ≤ trần": params[-1] is not None and params[-1] <= c.PARAM_CAP,
                      "latency ≤ trần": ms[-1] is None or ms[-1] <= cap_ms,
                      "params tăng theo K": all(b > a for a, b in zip(params, params[1:]) if a and b),
                      "latency tăng theo K": all(b > a for a, b in zip(ms_known, ms_known[1:])),
                      "PSNR tăng theo K": all(b > a for a, b in zip(psnr, psnr[1:])),
                      "A(SR)=y": consistency < c.CONSISTENCY_TOL}
            constraint_rows.append({"lr": method, "model": name, "selected": name == t.selected.get(method),
                                    "W": arch["width"], "block": arch["block"], "N": sum(arch["blocks"]), "alpha": arch["alpha"],
                                    "K": str(ks), "params_kmax": params[-1], "ms_kmax_fp32_256": ms[-1],
                                    **{f"psnr_K{k}": p for k, p in zip(ks, psnr)}, "psnr_kmax": psnr[-1],
                                    "lr_max_abs": consistency, **{f"ok: {k}": v for k, v in checks.items()},
                                    "all_ok": all(checks.values())})
    write_csv(ctx.compare / "constraints.csv", constraint_rows)
    parts += ["## Các biến thể NuSE-E và ràng buộc cứng (sau khi train)", md_table(
        ["LR", "model", "W", "block", "N", "alpha", "K", "params K max", "ms fp32 @256² K max", "PSNR theo K",
         "max|A(SR)−y|", "ràng buộc"],
        [[r["lr"], ("★ " if r["selected"] else "") + r["model"], r["W"], r["block"], r["N"], r["alpha"], r["K"],
          f"{r['params_kmax']:,}" if r["params_kmax"] else "—", fmt(r["ms_kmax_fp32_256"], 2),
          " / ".join(f"{r[k]:.3f}" for k in r if k.startswith("psnr_K")), f"{r['lr_max_abs']:.1e}",
          "đạt hết" if r["all_ok"] else "VI PHẠM: " + ", ".join(k[4:] for k in r if k.startswith("ok: ") and not r[k])]
         for r in constraint_rows])]
    warnings = [f"CẢNH BÁO: bản được chọn {r['model']} / {r['lr']} vi phạm ràng buộc, xem cột cuối."
                for r in constraint_rows if r["selected"] and not r["all_ok"]]

    rows = []
    for key, per in t.cross().items():
        row = [key]
        for m in c.METHODS:
            lab, r = per.get(m, (None, None))
            row.append("—" if r is None else f"{r['rgb_psnr']:.3f} / {r['rgb_ssim']:.4f} / {fmt(r.get('lpips'))} / "
                       f"{fmt(r.get('fid_patch256'), 1)}" + (f" ({lab.split(' ')[1]})" if lab.startswith("NuSE-E") else ""))
        rows.append(row)
    parts += ["## So sánh hai loại LR (DIV2K valid)", md_table(["model", *[f"{m}: PSNR / SSIM / LPIPS / FID" for m in c.METHODS]], rows)]

    if t.latency:
        rows = []
        for m in c.METHODS:
            for name in dict.fromkeys(n for n in ("NuSE-E", t.selected.get(m)) if n):
                for k in ctx.arch(m, name)["budgets"]:
                    for size in [f"{h}x{w}" for h, w in c.LATENCY_SIZES]:
                        for p in c.LATENCY_PRECISIONS:
                            pr, dc, tt = (t.lat(m, f"{name} K={k}", size, p, s) for s in ("predictor", "decoder + consistency", "total"))
                            if tt:
                                rows.append([m, name, k, size, p, fmt(pr, 3), fmt(dc, 3), fmt(tt, 3), f"{dc / tt:.0%}"])
        if rows:
            parts += ["## NuSE-E: phân rã latency (median ms, GPU)", md_table(
                ["LR", "model", "K", "kích thước LR", "precision", "predictor", "decoder + consistency", "total", "tỉ lệ decoder"], rows)]
    text = "# Kết quả NuSE-E\n\n" + "\n\n".join(parts + warnings) + "\n"
    (ctx.compare / "report.md").write_text(text)
    print(text)
    print("->", ctx.compare / "report.md")
