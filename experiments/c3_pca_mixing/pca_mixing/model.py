"""NuSE-E với basis học được: PCA mixing của Luân (nhánh exp/pca15-to-8-mixing) áp cho các exit lồng nhau.

Decoder của NuSEE dùng U (C, 16, 15) cố định; exit K lấy K cột đầu, tức K thành phần PCA đầu. Ở đây U' = U R với
R (C, 15, 15) trực chuẩn = QR(I + s·Δ) (orthonormal_columns, lấy từ code của Luân: dấu cột cố định, kiểm hạng).
Δ = 0 lúc khởi tạo nên U' = U: model cho output giống hệt NuSEE cùng trọng số, mọi khác biệt sau khi train đến từ R.
Exit K lấy K cột đầu của U', nên mỗi exit học không gian con K chiều của riêng nó thay vì K hướng phương sai lớn nhất.
U = Q V nằm trong null space nên U' cũng vậy: A(SR) = y giữ nguyên. Ở K = 15 (cả null space) R không đổi gì.

s = options["basis_lr_scale"] (mặc định 0.1). Adam đi ~lr mỗi bước trên Δ, tức ~s·lr trên I + s·Δ, nên với giao thức
chung (lr 1e-3 → 1e-6) R được học với 1e-4 → 1e-7, đúng lr basis trong run của Luân, mà không phải sửa trainer.

Loss (LR average pool): loss hệ số của Định lý 4 cộng năng lượng bị cắt Σ_K α_K Σ_{i>K} c'_i². NuSEE bỏ phần này vì nó
là hằng khi basis cố định; ở đây nó phụ thuộc R. Tổng bằng đúng Σ_K α_K · MSE ảnh ở exit K.
LR bicubic: NuSEE vốn train bằng MSE ảnh qua decoder, gradient tự đến R, giữ nguyên.

fuse() gộp R vào decoder cố định của NuSEE: suy luận không thêm tham số, GFLOPs hay latency.
"""
import torch
from torch import nn

from nuse_e.models.nuse_e import FixedDecoder, NuSEE


def orthonormal_columns(matrix):
    """Reduced QR with stable column signs and a rank check (Luân, nullspaceSR/pca.py)."""
    if matrix.ndim < 2 or matrix.shape[-2] < matrix.shape[-1]:
        raise ValueError("Expected a tall matrix")
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("Mixing parameters are non-finite")
    singular = torch.linalg.svdvals(matrix.detach())
    if (singular[..., -1] < 1e-6).any() or (singular[..., -1] / singular[..., 0].clamp_min(1e-30) < 1e-6).any():
        raise FloatingPointError("Mixing parameters are rank deficient")
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.where(torch.diagonal(r, dim1=-2, dim2=-1) < 0, -1, 1)
    return q * signs.unsqueeze(-2)


class MixingDecoder(nn.Module):
    """z = mu + U'_K c, U' = U R. Cùng giao diện FixedDecoder (forward, targets, basis), basis tính lại từ R."""

    def __init__(self, fixed, scale):
        super().__init__()
        self.channels, self.scale = fixed.channels, scale
        self.register_buffer("mean_block", fixed.mean_block.clone())          # (C,16)
        self.register_buffer("source_basis", fixed.basis.clone())             # U (C,16,d)
        d = fixed.basis.shape[-1]
        self.delta = nn.Parameter(torch.zeros(self.channels, d, d))

    def rotation(self):
        eye = torch.eye(self.delta.shape[-1], dtype=self.delta.dtype, device=self.delta.device)
        return orthonormal_columns(eye + self.scale * self.delta)

    @property
    def basis(self):
        return torch.einsum("cpd,cdk->cpk", self.source_basis, self.rotation())

    forward = FixedDecoder.forward
    targets = FixedDecoder.targets

    def fixed(self):
        return FixedDecoder(self.mean_block, self.basis.detach())


class MixingNuSEE(NuSEE):
    def __init__(self, stats, arch, channels=3):
        super().__init__(stats, arch, channels)
        self.decoder = MixingDecoder(self.decoder, (arch.get("options") or {}).get("basis_lr_scale", 0.1))

    def fuse(self):
        """Như NuSEE.fuse, rồi gộp R vào decoder cố định. Idempotent."""
        super().fuse()
        if isinstance(self.decoder, MixingDecoder):
            self.decoder = self.decoder.fixed()
        return self

    def _loss(self, chat, lr, hr, kmax, alpha):
        if self.method != "average_pool" or not isinstance(self.decoder, MixingDecoder):
            return super()._loss(chat, lr, hr, kmax, alpha)
        cstar = self.decoder.targets(self.residual_blocks(hr))              # c' = U'^T (x_b - mu): gradient đến R
        w = torch.zeros(kmax, device=lr.device)
        for k, a in zip(self.budgets, alpha):
            w[:k] += a
        err = (chat.reshape_as(cstar) - cstar).square()
        energy = cstar.square().sum((0, 1, 3, 4))
        tail = sum(a * energy[k:].sum() for k, a in zip(self.budgets, alpha))
        return ((w[None, None, :, None, None] * err).sum() + tail) / hr.numel()
