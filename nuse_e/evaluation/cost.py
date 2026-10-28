"""Chi phí: tham số, GFLOPs, latency GPU / CPU."""
import time

import torch

from ..models.nuse_e import NuSEE
from ..models.registry import budgets_of

def count_params(model, k=None):
    if isinstance(model, NuSEE):
        return model.params_for(k)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gflops(model, size, device, k=None):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.rand(1, 3, *size, device=device)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(x, k) if isinstance(model, NuSEE) else model(x)
    return counter.get_total_flops() / 1e9


@torch.no_grad()
def time_cuda(fns, precision, warmup=30, iters=300, rounds=10):
    """CUDA events, batch 1, interleaved over `rounds` so clock drift affects every entry alike."""
    times = {key: [] for key in fns}
    memory = {}
    with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
        for key, fn in fns.items():
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fn()
            torch.cuda.synchronize()
            memory[key] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        for _ in range(rounds):
            for key, fn in fns.items():
                events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                          for _ in range(max(1, iters // rounds))]
                for start, end in events:
                    start.record()
                    fn()
                    end.record()
                torch.cuda.synchronize()
                times[key] += [s.elapsed_time(e) for s, e in events]
    return times, memory


@torch.no_grad()
def time_cpu(fn, iters=20):
    fn()
    out = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return out


def stage_functions(label, model, x):
    """Entries to time: total for every model; NuSE-E also split into predictor and decoder + consistency."""
    fns = {}
    for k in budgets_of(model):
        name = label if k is None else f"{label} K={k}"
        if k is None:
            fns[(name, "total")] = (lambda m=model: m(x))
        else:
            coeffs = model.coefficients(x, k)
            fns[(name, "predictor")] = (lambda m=model, k=k: m.coefficients(x, k))
            fns[(name, "decoder + consistency")] = (lambda m=model, k=k, c=coeffs: m.consistency(m.detail(c, k), x))
            fns[(name, "total")] = (lambda m=model, k=k: m(x, k))
    return fns
