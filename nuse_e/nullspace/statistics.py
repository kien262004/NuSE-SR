"""Thống kê từ DIV2K train: PCA làm đối xứng D4 trên phần dư null space, Wiener và R² k-fold (stats.pt)."""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..core.constants import S
from ..data.cache import FlatImages
from ..data.loaders import RamCropLoader, TrainCrops
from .basis import d4_permutations, helmert, lr_windows, null_coordinates
from ..operators import degrade, make_operator

@torch.no_grad()
def fit_statistics(files, method, device, *, repeat=4, folds=5, batch=16, workers=2, seed=2026, log=print):
    """Pass 1: D4-symmetrised PCA of the null-space residual. Pass 2: Wiener weights + k-fold R^2.

    Training HR only (I8). FP64 accumulation. Returns a dict saved as stats.pt.
    """
    op = make_operator(method)
    q = helmert().to(device)
    d = 15 if method == "average_pool" else 16
    frame = q if method == "average_pool" else torch.eye(16, dtype=torch.float64, device=device)
    if isinstance(files, (str, Path)):                                  # flat uint8 cache (build_flat_cache)
        images = FlatImages(files)
        loader = RamCropLoader(images, batch, seed, 256, repeat, augment=False)
        names = [e["name"] for e in images.index]
    else:
        data = TrainCrops(files, 256, repeat, augment=False, seed=seed)
        loader = DataLoader(data, batch_size=batch, shuffle=False, num_workers=workers)
        names = [Path(p).name for p in files]

    def to_f64(hr):
        return hr.to(device, torch.float64) / (255 if hr.dtype == torch.uint8 else 1)
    n, total, outer = 0, None, None
    t0 = time.time()
    for step, (hr, _) in enumerate(loader):
        hr = to_f64(hr)
        u = null_coordinates(hr, method, op, q)                         # (C,N,d)
        if total is None:
            total = torch.zeros(u.shape[0], d, dtype=torch.float64, device=device)
            outer = torch.zeros(u.shape[0], d, d, dtype=torch.float64, device=device)
        total += u.sum(1)
        outer += torch.einsum("cni,cnj->cij", u, u)
        n += u.shape[1]
        if step % 50 == 0:
            log(f"[stats {method}] pass 1 batch {step + 1}/{len(loader)} ({time.time() - t0:.0f}s)")
    mean = total / n
    cov = outer / n - mean[:, :, None] * mean[:, None, :]
    cov = (cov + cov.transpose(1, 2)) / 2
    reps = [(frame.T @ p.to(device) @ frame) for p in d4_permutations()]
    cov_sym = sum(r @ cov @ r.T for r in reps) / len(reps)
    mean_sym = sum(mean @ r.T for r in reps) / len(reps)
    lam, vec = torch.linalg.eigh(cov_sym)
    lam, vec = lam.flip(-1).clamp_min(0), vec.flip(-1)                 # (C,d), (C,d,d) columns = v_i
    invariance = max(((r @ cov_sym @ r.T) - cov_sym).abs().max().item() for r in reps) / cov_sym.abs().max().item()

    xtx = torch.zeros(folds, 3, 10, 10, dtype=torch.float64, device=device)
    xty = torch.zeros(folds, 3, 10, d, dtype=torch.float64, device=device)
    ysum = torch.zeros(folds, 3, d, dtype=torch.float64, device=device)
    ysq = torch.zeros(folds, 3, d, dtype=torch.float64, device=device)
    cnt = torch.zeros(folds, dtype=torch.float64, device=device)
    t0 = time.time()
    for step, (hr, image_index) in enumerate(loader):
        hr = to_f64(hr)
        lr = degrade(hr, method)
        u = null_coordinates(hr, method, op, q)
        c = torch.einsum("cnd,cdk->cnk", u - mean_sym[:, None, :], vec)  # (C,N,d)
        x = lr_windows(lr)                                              # (C,N,10)
        per = (lr.shape[-2] * lr.shape[-1])
        fold_of = (image_index % folds).to(device).repeat_interleave(per)
        for f in range(folds):
            sel = fold_of == f
            if sel.any():
                xs, cs = x[:, sel], c[:, sel]
                xtx[f] += xs.transpose(1, 2) @ xs
                xty[f] += xs.transpose(1, 2) @ cs
                ysum[f] += cs.sum(1)
                ysq[f] += cs.square().sum(1)
                cnt[f] += sel.sum()
        if step % 50 == 0:
            log(f"[stats {method}] pass 2 batch {step + 1}/{len(loader)} ({time.time() - t0:.0f}s)")

    def solve(a, b):
        eps = 1e-8 * torch.diagonal(a, dim1=-2, dim2=-1).sum(-1)[..., None, None] / 10
        return torch.linalg.solve(a + eps * torch.eye(10, dtype=a.dtype, device=a.device), b)

    sse = torch.zeros(3, d, dtype=torch.float64, device=device)
    sst = torch.zeros(3, d, dtype=torch.float64, device=device)
    for f in range(folds):
        keep = [g for g in range(folds) if g != f]
        w = solve(xtx[keep].sum(0), xty[keep].sum(0))                  # (C,10,d)
        sse += ysq[f] - 2 * torch.einsum("cid,cid->cd", w, xty[f]) + torch.einsum("cid,cij,cjd->cd", w, xtx[f], w)
        sst += ysq[f] - ysum[f].square() / cnt[f].clamp_min(1)
    r2 = 1 - sse / sst.clamp_min(1e-30)
    wiener = solve(xtx.sum(0), xty.sum(0))                              # (C,10,d)
    return {"method": method, "scale": S, "d": d, "mean": mean_sym.cpu(), "components": vec.cpu(),
            "eigenvalues": lam.cpu(), "wiener": wiener.cpu(), "r2": r2.cpu(), "blocks_per_channel": n,
            "Q": helmert(), "d4_invariance": invariance, "n_train_images": len(names),
            "files": names, "torch_version": torch.__version__}
