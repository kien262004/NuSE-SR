"""Cách 3: basis học được trong null space (PCA mixing của Luân): exit K dùng K cột đầu của U·R, R trực chuẩn học được.

    model  MixingNuSEE, MixingDecoder, orthonormal_columns
"""
from .model import MixingDecoder, MixingNuSEE, orthonormal_columns

__all__ = ["MixingDecoder", "MixingNuSEE", "orthonormal_columns"]
