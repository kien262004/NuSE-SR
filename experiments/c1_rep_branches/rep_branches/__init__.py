"""Cách 1: conv ba nhánh cộng lại khi train, gộp thành một Conv3x3 khi suy luận (re-parameterisation).

    branches  ACBConv, ECBConv: conv ba nhánh và phép gộp
    model     RepBranchNuSEE: NuSE-E e80 với mỗi Conv3x3 của trunk là conv ba nhánh
"""
from .branches import ACBConv, ECBConv
from .model import BranchResBlock, RepBranchNuSEE

__all__ = ["ACBConv", "ECBConv", "BranchResBlock", "RepBranchNuSEE"]
