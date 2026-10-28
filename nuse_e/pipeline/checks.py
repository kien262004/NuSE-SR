"""Bước test (test chấp nhận) và screen (sàng tham số, latency trước khi train)."""
import statistics
import sys

import torch

from ..core.io import write_csv, write_json
from ..data.images import ValImages, as_float
from ..evaluation.cost import time_cuda
from ..models.blocks import ResBlock
from ..models.nuse_e import NuSEE
from ..models.registry import load_run
from ..nullspace.basis import helmert, lr_windows
from ..operators import degrade, make_operator

def step_test(ctx):
    """Test chấp nhận: phải pass hết trước khi train."""
    c = ctx.c
    results = []

    def check(name, ok, value=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {value}")
        results.append(bool(ok))

    q = helmert()
    check("T2 QᵀQ = I, 1ᵀQ = 0", (q.T @ q - torch.eye(15, dtype=q.dtype)).abs().max() < 1e-12 and q.sum(0).abs().max() < 1e-12)
    val_files = ctx.data()["val_files"]
    for method in c.METHODS:
        stats = torch.load(ctx.stats_dir / method / "stats.pt", map_location="cpu", weights_only=False)
        op = make_operator(method)
        hr = as_float(ValImages(val_files)[0][0], "cpu").double()
        lr_data = degrade(hr, method)
        if method == "bicubic":
            err = (op.forward_matrix(hr) - lr_data).abs().max()
            check(f"T1 [{method}] B (ma trận) = hàm tạo LR", err < 1e-6, f"{err:.1e}")
        if "d4_invariance" in stats:
            check(f"T3 [{method}] bất biến D4", stats["d4_invariance"] < 1e-12, f"{stats['d4_invariance']:.1e}")
        y = degrade(torch.rand(1, 3, 128, 160), method)
        for name in c.NUSE:
            arch = ctx.arch(method, name)
            tag = f"[{method} {name}]"
            model = NuSEE(stats, arch).eval()
            u = model.decoder.basis.double()
            check(f"T4 {tag} UᵀU = I", (torch.einsum("cpk,cpj->ckj", u, u) - torch.eye(u.shape[-1])).abs().max() < 1e-5)
            worst = 0.0
            for seed in range(3):
                torch.manual_seed(seed)
                net = NuSEE(stats, arch).eval()
                with torch.no_grad():
                    for p in net.parameters():
                        p.normal_(0, 1)
                    for k in net.budgets:
                        z = net.detail(net.coefficients(y, k), k)
                        worst = max(worst, ((op.forward(net(y, k)) - y).abs().max() / z.abs().max().clamp_min(1)).item())
            check(f"T5 {tag} A(SR)=y, trọng số ngẫu nhiên, sai số / max|z|", worst < 1e-6, f"{worst:.1e}")
            with torch.no_grad():
                k1 = model.budgets[min(1, len(model.budgets) - 1)]
                before = model(y, k1).clone()
                g1 = model.budgets.index(k1)
                for mod in list(model.stages[g1 + 1:]) + list(model.heads[g1 + 1:]):
                    for p in mod.parameters():
                        p.add_(torch.randn_like(p))
                check(f"T7 {tag} output ở K={k1} không phụ thuộc stage sau", torch.equal(before, model(y, k1)))
            model = NuSEE(stats, arch)
            W, blk = arch["width"], ResBlock.inference_params(arch["width"], arch["block"])
            formula = [28 * W + blk * sum(arch["blocks"][:g + 1]) + sum((3 * W + 3) * n + 30 * n * l for n, l in
                       zip(arch["groups"][:g + 1], arch["linear"][:g + 1])) for g in range(len(arch["budgets"]))]
            params = [model.params_for(k) for k in model.budgets]
            check(f"T8 {tag} số tham số suy luận theo công thức", params == formula, params)
            check(f"T9 {tag} mỗi ngân sách thêm ≥ 1 block", min([params[0]] + [b - a for a, b in zip(params, params[1:])]) >= blk)
            if arch["block"] == "rep":
                torch.manual_seed(0)
                net = NuSEE(stats, arch).eval()
                with torch.no_grad():
                    for p in net.parameters():
                        p.normal_(0, 0.3)
                    outs = [net(y, k).clone() for k in net.budgets]
                    net.fuse()
                    fused = [net(y, k) for k in net.budgets]
                    n_params = sum(p.numel() for p in net.parameters())
                err = max(((a - b).abs().max() / a.abs().max()).item() for a, b in zip(outs, fused))
                check(f"T11 {tag} gộp RepConv: output bằng nhau, tham số = công thức",
                      err < 1e-5 and n_params == params[-1], f"{err:.1e}, {n_params}")
            if arch["linear"][0]:
                lr = degrade(torch.rand(2, 3, 64, 64, dtype=torch.float64), method)
                g0 = arch["groups"][0]
                with torch.no_grad():
                    got = model.eval().coefficients(lr.float(), model.budgets[0]).double().reshape(2, 3, g0, 16, 16)
                x = lr_windows(lr)
                want = torch.einsum("cni,cik->cnk", x, stats["wiener"][:, :, :g0]).reshape(3, 2, 16, 16, g0).permute(1, 0, 4, 2, 3)
                rel = ((got - want).abs().max() / want.abs().max()).item()
                check(f"T10 {tag} lúc khởi tạo head = Wiener", rel < 1e-5, f"{rel:.1e}")
    if not all(results):
        sys.exit("Có test FAIL: dừng, không train.")
    print(f"Pass hết {len(results)} test.")


@torch.no_grad()
def quick_ms(fn, warmup=20, iters=100):
    t, _ = time_cuda({"x": fn}, "fp32", warmup, iters, 2)
    return statistics.median(t["x"])


def step_screen(ctx):
    """Tham số và latency (K lớn nhất, LR 256², fp32) của mọi biến thể; vượt trần thì không train."""
    c = ctx.c
    cap_ms, cap_src = c.LATENCY_CAP_MS, "số cũ (T4)"
    cuda = ctx.device.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True
        x256 = torch.rand(1, 3, 256, 256, device=ctx.device)
        if c.LATENCY_REF in ctx.old():
            ref_model = load_run(ctx.old()[c.LATENCY_REF])[0].to(ctx.device).eval()
            cap_ms, cap_src = quick_ms(lambda: ref_model(x256)), f"đo lại trên {torch.cuda.get_device_name(0)}"
            del ref_model
    print(f"Trần: tham số <= {c.PARAM_CAP:,} | latency @256² fp32 <= {cap_ms:.2f} ms ({cap_src})"
          + ("" if cuda else " | không có GPU: bỏ qua kiểm latency"))
    method = "average_pool" if "average_pool" in c.METHODS else c.METHODS[0]   # trunk giống nhau giữa hai loại LR
    stats = torch.load(ctx.stats_dir / method / "stats.pt", map_location="cpu", weights_only=False)
    rows = []
    for name in c.NUSE:
        arch = ctx.arch(method, name)
        model = NuSEE(stats, arch).fuse().to(ctx.device).eval()
        kmax = model.budgets[-1]
        params = model.params_for(kmax)
        ms = quick_ms(lambda m=model: m(x256, kmax)) if cuda else None
        ok = params <= c.PARAM_CAP and (ms is None or ms <= cap_ms)
        rows.append({"model": name, "params_kmax": params, "ms_256_fp32": ms, "admissible": ok})
        print(f"{'OK  ' if ok else 'LOẠI'} {name:14s} W={arch['width']} {arch['block']:5s} N={sum(arch['blocks'])} "
              f"K={arch['budgets']} | tham số K max {params:,} | {('%.2f ms' % ms) if ms else '—'}")
        del model
    write_csv(ctx.compare / "screen.csv", rows)
    write_json(ctx.compare / "screen.json", {"cap_ms": cap_ms, "cap_src": cap_src, "device": str(ctx.device), "rows": rows})
    if not next(r for r in rows if r["model"] == "NuSE-E")["admissible"]:
        sys.exit("Bản tham chiếu v2 phải qua sàng; nếu không thì trần đo sai.")
