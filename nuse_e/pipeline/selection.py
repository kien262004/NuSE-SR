"""Bước select: chọn biến thể tốt nhất mỗi loại LR."""
import torch

from ..core.io import write_csv, write_json

def step_select(ctx):
    """Chọn biến thể có PSNR K lớn nhất cao nhất (bằng nhau thì ít tham số hơn); bản v2 luôn giữ để đối chiếu."""
    c = ctx.c
    screen = {r["model"]: r for r in ctx.screen()["rows"]}
    selected, rows = {}, []
    for method in c.METHODS:
        cands = []
        for name in c.NUSE:
            if name not in screen or not screen[name]["admissible"] or not ctx.is_done(method, name):
                continue
            ck = torch.load(ctx.run_dir(method, name) / "best.pt", map_location="cpu", weights_only=False)
            rec = next(r for r in ck["history"] if r["epoch"] == ck["epoch"])
            arch = ck["arch"]
            # Checkpoint của bản v2 cũ không lưu width/block/alpha: dùng mặc định của NuSEE.
            row = {"lr": method, "model": name, "width": arch.get("width", 32), "block": arch.get("block", "plain"),
                   "blocks": sum(arch["blocks"]), "alpha": arch.get("alpha", "uniform"), "budgets": str(arch["budgets"]),
                   "best_epoch": ck["epoch"], "params_kmax": screen[name]["params_kmax"],
                   "psnr_mean_K": rec["validation"]["rgb_psnr"],
                   **{f"psnr_K{k}": rec["validation"][f"rgb_psnr_K{k}"] for k in arch["budgets"]}}
            rows.append(row)
            cands.append((rec["validation"][f"rgb_psnr_K{arch['budgets'][-1]}"], -row["params_kmax"], name))
        if cands:
            best = max(p for p, _, _ in cands)
            tied = [cand for cand in cands if cand[0] >= best - c.SELECT_TOL_DB]   # trong ngưỡng nhiễu: coi là hoà
            selected[method] = max(tied, key=lambda cand: (cand[1], cand[0]))[2]   # ít tham số hơn, rồi PSNR cao hơn
        else:
            selected[method] = None
        print(f"[{method}] chọn: {selected[method]} (chênh dưới {c.SELECT_TOL_DB} dB coi là hoà, lấy bản ít tham số hơn)")
        for cand in sorted(cands, reverse=True):
            print(f"   {cand[2]:14s} PSNR K max {cand[0]:.4f}  tham số {-cand[1]:,}")
    write_csv(ctx.compare / "variants.csv", rows)
    write_json(ctx.compare / "selected.json", selected)
