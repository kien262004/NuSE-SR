"""NuSE-E với trunk dùng conv ba nhánh. Sau `fuse()` model giống hệt e80: cùng tham số, FLOPs và latency."""
from functools import partial

from torch import nn
from torch.nn import functional as F

from nuse_e.models.nuse_e import NuSEE

from .branches import ACBConv, ECBConv

CONVS = {"acb": ACBConv, "ecb": ECBConv}


class BranchResBlock(nn.Module):
    """ResBlock của NuSE-E (Conv3x3 -> ReLU -> Conv3x3, skip identity), mỗi Conv3x3 là một conv ba nhánh."""

    def __init__(self, width, conv):
        super().__init__()
        self.conv1 = conv(width)
        self.conv2 = conv(width)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))

    def fuse(self):
        for name in ("conv1", "conv2"):
            conv = getattr(self, name)
            if not isinstance(conv, nn.Conv2d):
                setattr(self, name, conv.fuse())


class RepBranchNuSEE(NuSEE):
    """arch["block"] chọn kiểu conv ba nhánh ("acb", "ecb"); arch["options"]["expand"] cho ECB (mặc định 2).

    Stage, head, exit và loss giữ nguyên NuSEE; chỉ thay block của trunk. `params_for` của NuSEE đã tính block ở dạng
    gộp (bằng Conv3x3 thường), nên các ràng buộc tham số và latency là của model sau khi gộp.
    """

    def __init__(self, stats, arch, channels=3):
        super().__init__(stats, arch, channels)
        conv = CONVS[arch["block"]]
        if arch["block"] == "ecb":
            conv = partial(conv, expand=arch.get("options", {}).get("expand", 2))
        self.stages = nn.ModuleList(nn.Sequential(*[BranchResBlock(self.width, conv) for _ in range(n)])
                                    for n in arch["blocks"])
