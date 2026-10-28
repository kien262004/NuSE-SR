"""LPIPS (AlexNet) và FID trên patch 256×256."""
import torch
from torch import nn

class LPIPSMetric(nn.Module):
    """LPIPS v0.1 AlexNet, clamp to [0,1] then [-1,1]."""

    def __init__(self):
        super().__init__()
        import lpips
        with torch.random.fork_rng(devices=[]):
            self.net = lpips.LPIPS(net="alex", version="0.1", verbose=False)
        self.net.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, sr, hr):
        return self.net(sr.float().clamp(0, 1) * 2 - 1, hr.float().clamp(0, 1) * 2 - 1).flatten()


class PatchFID:
    """FID on non-overlapping 256x256 patches (Inception-v3 pool 2048), real features kept across models."""

    def __init__(self, device, patch=256):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.metric = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
        self.patch, self.has_real = patch, False

    def patches(self, x):
        p = self.patch
        h, w = x.shape[-2] // p * p, x.shape[-1] // p * p
        if h == 0 or w == 0:
            return x.clamp(0, 1)
        x = x[..., :h, :w].clamp(0, 1)
        return x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, p, p)

    def update(self, image, real):
        for chunk in self.patches(image).split(32):
            self.metric.update(chunk.float(), real=real)

    def compute_and_reset(self):
        value = self.metric.compute().item()
        self.metric.reset()
        return value
