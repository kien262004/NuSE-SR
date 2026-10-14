"""Blockwise null-space PCA for the average-pooling A in Wang et al., AAAI 2023.

Reference operator:
    A  = avg_pool2d(kernel_size=scale, stride=scale)
    A+ = nearest-neighbor repetition (repeat_interleave)
    P  = I - A+ A

For every non-overlapping scale-by-scale block and each color channel:
    P_b x_b = x_b - mean(x_b) * ones.

This is a working research prototype, NOT a reimplementation of the paper's GAN.
Requirements: PyTorch >= 2.0. Demonstration needs no external dataset.
Use HR crops whose spatial dimensions are divisible by scale.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MeanSRLinearOperator:
    """Average-pooling downsampler A and Moore-Penrose pseudoinverse A+.

    NOTE: exact only for this nonoverlapping block-average forward model,
    not for bicubic/Gaussian-blur downsampling.
    """
    def __init__(self, scale: int = 4):
        if scale < 2:
            raise ValueError("scale must be >= 2")
        self.scale = int(scale)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scale
        if x.ndim != 4 or x.shape[-2] % s or x.shape[-1] % s:
            raise ValueError("Expected BCHW image with H and W divisible by scale")
        return F.avg_pool2d(x, kernel_size=s, stride=s)

    def Ap(self, y: torch.Tensor) -> torch.Tensor:
        s = self.scale
        return y.repeat_interleave(s, -2).repeat_interleave(s, -1)

    def P(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.Ap(self.A(x))


def to_block_patches(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """B,C,sH,sW -> B,C,s*s,H,W in PixelShuffle-compatible order."""
    b, c, h, w = hr.shape
    q = scale * scale
    return F.pixel_unshuffle(hr, scale).reshape(b, c, q, h // scale, w // scale)


class BlockNullSpacePCA(nn.Module):
    """One PCA basis per color channel, shared across LR spatial positions.

    basis shape: (C, rank, scale**2).
    mu shape:    (C, scale**2), constrained to sum zero per channel.
    """
    def __init__(self, basis: torch.Tensor, mu: torch.Tensor, scale: int):
        super().__init__()
        channels, rank, q = basis.shape
        if q != scale * scale or mu.shape != (channels, q):
            raise ValueError("Wrong shapes for PCA basis/mean")
        if rank > q - 1:
            raise ValueError("For block-mean A, nullity per channel = scale**2-1")
        self.scale = scale
        self.channels = channels
        self.rank = rank
        self.register_buffer("basis", basis.float().contiguous())
        self.register_buffer("mu", mu.float().contiguous())

    @classmethod
    @torch.no_grad()
    def fit(cls, hr_batches, scale=4, rank=8):
        """Stream covariance on null-space HR patches without building P.

        hr_batches is an iterable of B,C,H,W FLOAT tensors (pixel range [0,1]).
        Returns (fitted module, all_eigenvalues[C,scale**2] in descending order).
        """
        q = scale**2
        if not 1 <= rank <= q - 1:
            raise ValueError(f"rank must lie in [1,{q-1}]")
        sum_z, sum_zz = None, None
        total = 0
        for hr in hr_batches:
            if isinstance(hr, (tuple, list)):
                hr = hr[0]
            if not isinstance(hr, torch.Tensor) or hr.ndim != 4:
                raise ValueError("hr_batches must yield tensors shaped B,C,H,W")
            patches = to_block_patches(hr.detach().cpu().to(torch.float64), scale)
            c = patches.shape[1]
            # C, num_patches, scale**2
            flat = patches.permute(1, 0, 3, 4, 2).reshape(c, -1, q)
            z = flat - flat.mean(dim=-1, keepdim=True)
            if sum_z is None:
                sum_z = torch.zeros((c, q), dtype=torch.float64)
                sum_zz = torch.zeros((c, q, q), dtype=torch.float64)
            elif c != sum_z.shape[0]:
                raise ValueError("All training batches must have same channels")
            sum_z += z.sum(dim=1)
            sum_zz += torch.einsum("cni,cnj->cij", z, z)
            total += flat.shape[1]
        if total < 2:
            raise ValueError("Need at least two training patches")
        mu = sum_z / total
        mu = mu - mu.mean(dim=-1, keepdim=True)
        cov = sum_zz / total - torch.einsum("ci,cj->cij", mu, mu)
        projector = torch.eye(q, dtype=torch.float64) - torch.ones((q,q), dtype=torch.float64)/q
        cov = projector @ cov @ projector
        cov = (cov + cov.transpose(-1, -2)) / 2
        eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
        eigvals = eigvals.flip(-1).clamp_min(0)
        basis = eigvecs[..., -rank:].flip(-1).transpose(-1,-2) # C,rank,q
        # Extra numerical safety: enforce exact zero-block-mean and orthonormality.
        basis = basis - basis.mean(dim=-1, keepdim=True)
        basis_q, _ = torch.linalg.qr(basis.transpose(-1, -2), mode='reduced')
        basis = basis_q.transpose(-1, -2)
        return cls(basis, mu, scale), eigvals

    def decode_null(self, coeffs: torch.Tensor) -> torch.Tensor:
        """B,C*rank,H,W -> B,C,H*scale,W*scale, each block mean zero."""
        b, out_ch, h, w = coeffs.shape
        if out_ch != self.channels * self.rank:
            raise ValueError(f"Expected {self.channels*self.rank} channels, got {out_ch}")
        coeffs = coeffs.reshape(b, self.channels, self.rank, h, w)
        # Output BCHW rearrangement precisely matches F.pixel_shuffle's layout.
        patches = (torch.einsum('bckhw,ckp->bcphw', coeffs, self.basis)
                   + self.mu[None, :, :, None, None])
        patches = patches.reshape(b, self.channels * self.scale**2, h, w)
        return F.pixel_shuffle(patches, self.scale)

    def encode_hr(self, hr: torch.Tensor) -> torch.Tensor:
        """PCA coefficients of the true HR's null-space component."""
        op = MeanSRLinearOperator(self.scale)
        z = op.P(hr)
        patches = to_block_patches(z, self.scale)
        patches = patches - self.mu[None, :, :, None, None]
        coeff = torch.einsum('bcphw,ckp->bckhw', patches, self.basis)
        return coeff.flatten(1, 2)

    def forward(self, lr: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        op = MeanSRLinearOperator(self.scale)
        return op.Ap(lr) + self.decode_null(coeffs)


class SmallCoefficientCNN(nn.Module):
    """Minimal LR-domain backbone. Replace with RLFN / SPAN / PLKSR trunk."""
    def __init__(self, in_channels: int, out_channels: int, width: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class NullSpacePCASR(nn.Module):
    def __init__(self, pca: BlockNullSpacePCA, width=32):
        super().__init__()
        self.pca = pca
        self.predictor = SmallCoefficientCNN(pca.channels, pca.channels * pca.rank, width)

    def forward(self, lr):
        coeff = self.predictor(lr)
        return self.pca(lr, coeff)


def demonstration():
    torch.manual_seed(2026)
    scale, rank, c = 4, 8, 3
    op = MeanSRLinearOperator(scale)
    # Structured synthetic HR samples for a self-contained smoke test.
    hr_batches = []
    for _ in range(4):
        lr_seed = torch.rand(3, c, 16, 16)
        low = F.interpolate(lr_seed, size=(64,64), mode='bilinear', align_corners=False)
        yy = torch.arange(64)[None,None,:,None]
        xx = torch.arange(64)[None,None,None,:]
        detail = (0.03 * torch.sin(0.31 * xx + 0.21 * yy)).float()
        hr_batches.append((low + detail).clamp(0,1))

    pca, evals = BlockNullSpacePCA.fit(hr_batches, scale=scale, rank=rank)
    model = NullSpacePCASR(pca, width=16)
    hr = hr_batches[0]
    lr = op.A(hr)
    pred = model(lr)

    print('Image shapes LR, HR:', tuple(lr.shape), tuple(pred.shape))
    print('PCA basis shape:', tuple(pca.basis.shape))
    print('Null basis mean max abs:', pca.basis.mean(-1).abs().max().item())
    print('Consistency max |A(pred)-LR|:', (op.A(pred)-lr).abs().max().item())
    assert torch.allclose(op.A(pred), lr, atol=5e-7, rtol=1e-6)

    # Verify orthogonal decomposition for arbitrary CNN coefficients.
    oracle_coeff = pca.encode_hr(hr)
    oracle_sr = pca(lr, oracle_coeff)
    trunc_error = (hr - oracle_sr).square().sum()
    prediction_error = (oracle_coeff - model.predictor(lr)).square().sum()
    total_error = (hr - pred).square().sum()
    print('Total error:', total_error.item())
    print('Truncation + coeff-prediction error:',
          (trunc_error + prediction_error).item())
    assert torch.allclose(total_error, trunc_error + prediction_error,
                          rtol=2e-5, atol=1e-4)

    # One supervised optimization step (use DIV2K loaders for real experiments).
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = F.mse_loss(model(lr), hr)
    loss.backward()
    optimizer.step()
    print('One train step MSE:', loss.item())
    print('Consistency after optimizer step:',
          (op.A(model(lr))-lr).abs().max().item())
    # Explained variance for each RGB channel at chosen rank.
    explained = (evals[:,:rank].sum(-1) / evals.sum(-1).clamp_min(1e-15))
    print('Explained variance by channel:', explained.tolist())


if __name__ == '__main__':
    demonstration()
