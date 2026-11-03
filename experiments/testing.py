"""Phần dùng chung cho test của các thí nghiệm (chạy không cần dữ liệu).

Mỗi `c*/test_*.py` thêm gốc repo và thư mục thí nghiệm vào sys.path rồi `from experiments.testing import ...`.
"""
import importlib.util
from pathlib import Path

import torch

from nuse_e.nullspace.basis import helmert

# arch_e80.json: exit K = 5 / 9 / 15, nhóm 5 / 4 / 6, nhánh tuyến tính L_g chỉ ở nhóm 1
E80_ARCH = {"budgets": [5, 9, 15], "groups": [5, 4, 6], "linear": [True, False, False],
            "method": "average_pool", "alpha": "uniform"}


def load_experiment(folder):
    """Module experiment.py của một thí nghiệm (IMPL, VARIANTS, ...)."""
    spec = importlib.util.spec_from_file_location(f"{Path(folder).name}_experiment", Path(folder) / "experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_stats(channels=3, d=15, seed=0):
    """Thống kê PCA giả (LR average pool), đủ khoá để suy kiến trúc và dựng NuSEE."""
    g = torch.Generator().manual_seed(seed)
    return {"method": "average_pool", "Q": helmert(),
            "components": torch.linalg.qr(torch.randn(channels, d, d, generator=g, dtype=torch.float64))[0],
            "mean": torch.randn(channels, d, generator=g, dtype=torch.float64) * 1e-2,
            "eigenvalues": torch.logspace(-2, -5, d, dtype=torch.float64).repeat(channels, 1),
            "wiener": torch.randn(channels, 10, d, generator=g, dtype=torch.float64) * 1e-2,
            "r2": torch.full((channels, d), 0.2, dtype=torch.float64)}


def e80_arch(blocks=(5, 1, 1), width=32, block="plain", impl=None):
    """Arch dạng e80; blocks / width nhỏ để test chạy nhanh. `impl`: lớp NuSEE của thí nghiệm."""
    arch = dict(E80_ARCH, blocks=list(blocks), width=width, block=block)
    if impl:
        arch["impl"], arch["options"] = impl, {}
    return arch


def randomize(modules, std=0.3, seed=0):
    """Gán N(0, std) cho mọi tham số, để so sánh không bị che bởi khởi tạo 0 / đơn vị."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for module in modules:
            for p in module.parameters():
                p.normal_(0, std)
