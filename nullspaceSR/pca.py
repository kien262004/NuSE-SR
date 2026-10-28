"""Offline PCA in analytic null coordinates; fixed synthesis at inference."""
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn

from .patches import from_blocks, to_blocks


def block_null_basis(scale: int, *, dtype=torch.float64) -> Tensor:
    """Helmert Q: (s*s,s*s-1), Q.T Q=I and ones.T Q=0."""
    if not isinstance(scale, int) or isinstance(scale, bool) or scale < 2:
        raise ValueError("scale must be an integer >= 2")
    q = scale**2
    basis = torch.zeros(q, q - 1, dtype=dtype)
    for j in range(1, q):
        norm = (j * (j + 1)) ** 0.5
        basis[:j, j - 1] = 1 / norm
        basis[j, j - 1] = -j / norm
    return basis


def orthonormal_columns(matrix: Tensor) -> Tensor:
    """Reduced QR with stable column signs and an explicit rank check."""
    if matrix.ndim < 2 or matrix.shape[-2] < matrix.shape[-1]:
        raise ValueError("Expected a tall matrix")
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("Mixing parameters are non-finite")
    singular = torch.linalg.svdvals(matrix.detach())
    if (singular[..., -1] < 1e-6).any() or (
        singular[..., -1] / singular[..., 0].clamp_min(1e-30) < 1e-6
    ).any():
        raise FloatingPointError("Mixing parameters are rank deficient")
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.where(torch.diagonal(r, dim1=-2, dim2=-1) < 0, -1, 1)
    return q * signs.unsqueeze(-2)


class BlockNullSpacePCA(nn.Module):
    """Per-channel frozen PCA shared by spatial blocks.

    Q: (q,d), mean: (C,d), components: (C,d,k), d=q-1.
    Synthesis Q(mean + V c) lies in ker(block-average A) for every c.
    """

    def __init__(self, scale: int, mean: Tensor, components: Tensor,
                 eigenvalues: Tensor | None = None) -> None:
        super().__init__()
        if not mean.is_floating_point() or mean.dtype != components.dtype:
            raise ValueError("mean/components must have the same floating dtype")
        Q = block_null_basis(scale, dtype=mean.dtype)
        d = scale**2 - 1
        if mean.ndim != 2 or mean.shape[1] != d or mean.shape[0] < 1:
            raise ValueError("mean must have shape (channels, scale**2-1)")
        if components.ndim != 3 or components.shape[:2] != mean.shape:
            raise ValueError("components must have shape (channels, scale**2-1, rank)")
        rank = components.shape[-1]
        if not 1 <= rank <= d:
            raise ValueError("rank must lie in [1, scale**2-1]")
        if mean.device != components.device:
            raise ValueError("mean/components must be on the same device")
        if not torch.isfinite(mean).all() or not torch.isfinite(components).all():
            raise ValueError("PCA buffers must be finite")
        gram = components.transpose(-1, -2) @ components
        identity = torch.eye(rank, dtype=components.dtype, device=components.device)
        if not torch.allclose(gram, identity.expand_as(gram), atol=1e-5, rtol=1e-5):
            raise ValueError("PCA components must be orthonormal")
        if eigenvalues is None:
            eigenvalues = mean.new_empty((mean.shape[0], 0))
        elif eigenvalues.shape != mean.shape or not torch.isfinite(eigenvalues).all():
            raise ValueError("eigenvalues must be finite with shape (channels, scale**2-1)")
        self.scale, self.channels, self.rank = scale, mean.shape[0], rank
        self.register_buffer("Q", Q.to(mean.device))
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("components", components.detach().clone())
        self.register_buffer("eigenvalues", eigenvalues.detach().to(mean).clone())
        # Precompute QV and Qm once; inference synthesis scales with k.
        self.register_buffer("synthesis_basis", torch.einsum("pd,cdk->cpk", self.Q, self.components))
        self.register_buffer("block_mean", torch.einsum("pd,cd->cp", self.Q, self.mean))

    @property
    def basis(self) -> Tensor:
        """Image-block basis (C,q,k) for offline diagnostics."""
        return self.synthesis_basis

    @classmethod
    @torch.no_grad()
    def fit(cls, hr_batches: Iterable[Tensor], scale: int = 4,
            rank: int = 8) -> "BlockNullSpacePCA":
        """TRAIN HR only. Stream CPU FP64 moments; return FP32 buffers.

        Zero-variance directions are still valid null-space directions.
        At full rank the decoder spans the complete local null space.
        """
        Q = block_null_basis(scale)
        d = Q.shape[1]
        if not isinstance(rank, int) or not 1 <= rank <= d:
            raise ValueError(f"rank must be an integer in [1,{d}]")
        sums = products = None
        count = 0
        for hr in hr_batches:
            if not isinstance(hr, Tensor) or not hr.is_floating_point():
                raise ValueError("hr_batches must yield floating BCHW tensors")
            if not torch.isfinite(hr).all():
                raise ValueError("Training HR contains non-finite values")
            blocks = to_blocks(hr.detach().to(device="cpu", dtype=torch.float64), scale)
            flat = blocks.permute(1, 0, 3, 4, 2).reshape(blocks.shape[1], -1, scale**2)
            coords = flat @ Q
            if sums is None:
                sums = torch.zeros(coords.shape[0], d, dtype=torch.float64)
                products = torch.zeros(coords.shape[0], d, d, dtype=torch.float64)
            if coords.shape[0] != sums.shape[0]:
                raise ValueError("Training batches must have the same channels")
            sums += coords.sum(1)
            products += torch.einsum("cni,cnj->cij", coords, coords)
            count += coords.shape[1]
        if count < 2:
            raise ValueError("PCA fitting requires at least two blocks per channel")
        mean = sums / count
        cov = products / count - mean.unsqueeze(-1) * mean.unsqueeze(-2)
        cov = (cov + cov.transpose(-1, -2)) / 2
        eigenvalues, vectors = torch.linalg.eigh(cov)
        return cls(scale, mean.float(), vectors.flip(-1)[..., :rank].float(),
                   eigenvalues.flip(-1).clamp_min(0).float())

    def encode_hr(self, hr: Tensor) -> Tensor:
        """Oracle coefficients from HR; training/evaluation only."""
        blocks = to_blocks(hr, self.scale)
        if blocks.shape[1] != self.channels:
            raise ValueError("HR channels do not match PCA")
        centered = blocks - self.block_mean[None, :, :, None, None]
        coeffs = torch.einsum("bcphw,cpk->bckhw", centered, self.synthesis_basis)
        return coeffs.flatten(1, 2)

    def decode_null(self, coeffs: Tensor) -> Tensor:
        """B,C*k,H,W -> B,C,sH,sW with zero block averages."""
        if coeffs.ndim != 4 or coeffs.shape[1] != self.channels * self.rank:
            raise ValueError(f"Expected BCHW coefficients with {self.channels * self.rank} channels")
        b, _, h, w = coeffs.shape
        coeffs = coeffs.reshape(b, self.channels, self.rank, h, w)
        blocks = torch.einsum("bckhw,cpk->bcphw", coeffs, self.synthesis_basis)
        blocks = blocks + self.block_mean[None, :, :, None, None]
        return from_blocks(blocks, self.scale)

    def forward(self, coeffs: Tensor) -> Tensor:
        return self.decode_null(coeffs)

    def save(self, path: str | Path) -> None:
        torch.save({"scale": self.scale, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "BlockNullSpacePCA":
        data = torch.load(path, map_location="cpu", weights_only=True)
        state = data["state_dict"]
        eigenvalues = state["eigenvalues"]
        result = cls(data["scale"], state["mean"], state["components"],
                     eigenvalues if eigenvalues.numel() else None)
        result.load_state_dict(state)
        return result


class PCAMixingBasis(nn.Module):
    """Learn an orthonormal rank-k basis inside a fixed PCA source span."""

    is_learnable = True
    basis_mode = "pca_mixing"

    def __init__(self, scale: int, mean: Tensor, source_components: Tensor, *,
                 rank: int = 8, eigenvalues: Tensor | None = None) -> None:
        super().__init__()
        if mean.ndim != 2 or source_components.ndim != 3 or source_components.shape[:2] != mean.shape:
            raise ValueError("mean/source_components shapes are incompatible")
        d = scale**2 - 1
        source_rank = source_components.shape[-1]
        if mean.shape[-1] != d or not 1 <= rank <= source_rank <= d:
            raise ValueError("Expected 1 <= rank <= source_rank <= scale**2-1")
        if mean.dtype != source_components.dtype or mean.device != source_components.device:
            raise ValueError("mean/source_components dtype and device must match")
        if not torch.isfinite(mean).all() or not torch.isfinite(source_components).all():
            raise ValueError("PCA buffers must be finite")
        gram = source_components.transpose(-1, -2) @ source_components
        identity = torch.eye(source_rank, dtype=gram.dtype, device=gram.device)
        if not torch.allclose(gram, identity.expand_as(gram), atol=1e-5, rtol=1e-5):
            raise ValueError("source_components must be orthonormal")
        if eigenvalues is None:
            eigenvalues = mean.new_empty((mean.shape[0], 0))
        elif eigenvalues.shape != mean.shape or not torch.isfinite(eigenvalues).all():
            raise ValueError("eigenvalues must have shape (channels, scale**2-1) and be finite")
        self.scale, self.channels, self.rank = scale, mean.shape[0], rank
        self.source_rank = source_rank
        self.register_buffer("Q", block_null_basis(scale, dtype=mean.dtype).to(mean.device))
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("source_components", source_components.detach().clone())
        self.register_buffer("eigenvalues", eigenvalues.detach().clone())
        self.register_buffer("block_mean", torch.einsum("pd,cd->cp", self.Q, self.mean))
        raw = mean.new_zeros((self.channels, rank, source_rank))
        raw[:, :, :rank] = torch.eye(rank, dtype=mean.dtype, device=mean.device)
        self.raw_mixing = nn.Parameter(raw)
        self.register_buffer("initial_basis", torch.einsum(
            "pd,cdk->cpk", self.Q, source_components[..., :rank]).detach().clone())

    @classmethod
    def from_pca(cls, decoder: BlockNullSpacePCA, *, rank: int,
                 source_rank: int = 15) -> "PCAMixingBasis":
        if source_rank > decoder.rank:
            raise ValueError("PCA artifact does not contain requested source rank")
        return cls(decoder.scale, decoder.mean, decoder.components[..., :source_rank],
                   rank=rank, eigenvalues=decoder.eigenvalues if decoder.eigenvalues.numel() else None)

    def effective_components(self) -> Tensor:
        mixing = orthonormal_columns(self.raw_mixing.transpose(-1, -2))
        return torch.einsum("cdm,cmk->cdk", self.source_components, mixing)

    def effective_basis(self) -> Tensor:
        return torch.einsum("pd,cdk->cpk", self.Q, self.effective_components())

    @property
    def basis(self) -> Tensor:
        return self.effective_basis()

    def encode_hr(self, hr: Tensor) -> Tensor:
        blocks = to_blocks(hr, self.scale)
        if blocks.shape[1] != self.channels:
            raise ValueError("HR channels do not match decoder")
        centered = blocks - self.block_mean[None, :, :, None, None]
        return torch.einsum("bcphw,cpk->bckhw", centered, self.effective_basis()).flatten(1, 2)

    def decode_null(self, coeffs: Tensor) -> Tensor:
        if coeffs.ndim != 4 or coeffs.shape[1] != self.channels * self.rank:
            raise ValueError(f"Expected {self.channels * self.rank} coefficient channels")
        b, _, h, w = coeffs.shape
        coeffs = coeffs.reshape(b, self.channels, self.rank, h, w)
        blocks = torch.einsum("bckhw,cpk->bcphw", coeffs, self.effective_basis())
        return from_blocks(blocks + self.block_mean[None, :, :, None, None], self.scale)

    def forward(self, coeffs: Tensor) -> Tensor:
        return self.decode_null(coeffs)
