"""Model.

    blocks     RepConv, ResBlock: khối của trunk
    nuse_e     NuSEE = trunk theo stage + head hệ số + decoder PCA cố định + nhất quán dữ liệu
    baselines  SPANV2, DISP, PLKSR-Tiny và bản cắm NuSE k=8 của thí nghiệm cũ
    registry   build_model (tạo theo tên), make_nuse (NuSEE hoặc lớp con của thí nghiệm), load_run, budgets_of
"""
from .nuse_e import NuSEE
from .registry import budgets_of, build_model, load_run, make_nuse

__all__ = ["NuSEE", "budgets_of", "build_model", "load_run", "make_nuse"]
