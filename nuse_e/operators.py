"""Toán tử đo A (tạo LR từ HR) và giả nghịch đảo A⁺.

average_pool: trung bình ô 4×4, A⁺ = lặp điểm. bicubic: bicubic ×4 có antialias, A⁺ dạng băng (Định lý 5–6)."""
import torch
from torch.nn import functional as F

from .core.constants import S

def avg_down(x):
    return F.avg_pool2d(x, S)


def bicubic_down(x):
    """PyTorch antialiased bicubic x1/4 (a = -0.5 inside the image, same as MATLAB imresize)."""
    return F.interpolate(x, scale_factor=1 / S, mode="bicubic", align_corners=False, antialias=True)


def degrade(x, method):
    return avg_down(x) if method == "average_pool" else bicubic_down(x)


class AvgPoolOp:
    name = "average_pool"

    def forward(self, x):
        return avg_down(x)

    def pinv(self, y):
        return y.repeat_interleave(S, -2).repeat_interleave(S, -1)


class BicubicOp:
    """B(x) = M_h x M_w^T (Thm. 5); exact separable pseudo-inverse B+ = M_h^+ (.) M_w^+T.

    M is extracted from bicubic_down itself (impulse responses), so B is exactly the
    function that generates the LR data. Matrices are cached per (H, W, device, dtype).
    """
    name = "bicubic"

    def __init__(self):
        self.cpu_cache = {}            # (H, W) -> FP64 matrices on CPU; one entry per image size

    @staticmethod
    def matrix(n):
        eye = torch.eye(n, dtype=torch.float64).reshape(n, 1, 1, n).expand(n, 1, S, n)
        out = bicubic_down(eye)                                        # (n,1,1,n/S)
        return out.reshape(n, n // S).T.contiguous()                   # (n/S, n)

    def mats(self, h, w, device, dtype):
        if (h, w) not in self.cpu_cache:
            mh, mw = self.matrix(h), self.matrix(w)
            self.cpu_cache[(h, w)] = tuple(t.float() for t in (mh, mw, torch.linalg.pinv(mh), torch.linalg.pinv(mw)))
        key = (h, w, str(device), dtype)
        if getattr(self, "_last_key", None) != key:
            self._last = tuple(t.to(device=device, dtype=dtype) for t in self.cpu_cache[(h, w)])
            self._last_key = key
        return self._last

    def forward(self, x):
        return bicubic_down(x)

    def forward_matrix(self, x):
        mh, mw, _, _ = self.mats(x.shape[-2], x.shape[-1], x.device, x.dtype)
        return mh @ x @ mw.T

    def pinv(self, y):
        _, _, ph, pw = self.mats(y.shape[-2] * S, y.shape[-1] * S, y.device, y.dtype)
        return ph @ y @ pw.T


def make_operator(method):
    return AvgPoolOp() if method == "average_pool" else BicubicOp()
