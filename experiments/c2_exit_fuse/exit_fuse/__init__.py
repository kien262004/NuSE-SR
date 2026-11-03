"""Cách 2: head của mỗi exit đọc concat các stage đã chạy, fuse bằng Conv1x1 (kiểu IMDN / RFB).

    model  ExitFuseNuSEE
"""
from .model import ExitFuseNuSEE

__all__ = ["ExitFuseNuSEE"]
