"""Cơ sở Helmert của null space, tách ô 4×4, đối xứng D4, cửa sổ LR 3×3."""
import math

import torch
from torch.nn import functional as F

from ..core.constants import S

def helmert(scale=S, dtype=torch.float64):
    """Q (q, q-1): Q^T Q = I, 1^T Q = 0."""
    q = scale ** 2
    basis = torch.zeros(q, q - 1, dtype=dtype)
    for j in range(1, q):
        norm = math.sqrt(j * (j + 1))
        basis[:j, j - 1] = 1 / norm
        basis[j, j - 1] = -j / norm
    return basis


def to_blocks(x):
    """(B,C,H,W) -> (B,C,16,H/4,W/4), pixel index p = dy*4 + dx."""
    b, c, h, w = x.shape
    return F.pixel_unshuffle(x, S).reshape(b, c, S * S, h // S, w // S)


def d4_permutations():
    idx = torch.arange(S * S).reshape(S, S)
    mats = []
    for turns in range(4):
        for flip in (False, True):
            perm = torch.rot90(idx, turns, (0, 1))
            perm = perm.flip(1) if flip else perm
            p = torch.zeros(S * S, S * S, dtype=torch.float64)
            p[torch.arange(S * S), perm.flatten()] = 1
            mats.append(p)
    return mats


def null_coordinates(hr, method, op, q):
    """Per-channel null-space coordinates u of every block: (C, N, d)."""
    if method == "average_pool":
        blocks = to_blocks(hr)                                         # (B,C,16,h,w)
        u = torch.einsum("bcphw,pd->cbhwd", blocks, q)
    else:
        r = hr - op.pinv(op.forward(hr))
        u = to_blocks(r).permute(1, 0, 3, 4, 2)                        # (C,B,h,w,16)
    return u.reshape(u.shape[0], -1, u.shape[-1])


def lr_windows(lr):
    """Per-channel 3x3 replicate-padded LR windows + 1: (C, N, 10), row-major taps."""
    b, c, h, w = lr.shape
    x = F.pad(lr.reshape(b * c, 1, h, w), (1, 1, 1, 1), mode="replicate")
    win = F.unfold(x, 3).reshape(b, c, 9, h * w).permute(1, 0, 3, 2).reshape(c, b * h * w, 9)
    return torch.cat([win, torch.ones_like(win[..., :1])], -1)
