"""NuSE-E với concat + fuse ở mỗi exit: head nhóm g đọc Conv1x1(concat(f_1, ..., f_g)) thay vì chỉ f_g.

f_j là output của stage j. Trunk giữ nguyên NuSEE (stage j + 1 vẫn chạy tiếp từ f_j), nên exit g chỉ dùng stage
1..g và mixer 1..g: chi phí vẫn tăng theo K và exit nhỏ không phụ thuộc phần sau (test T7 của pipeline kiểm cả mixer).

Mixer khởi tạo để chọn đúng f_g (ma trận đơn vị ở khối của f_g, còn lại 0), nên lúc bắt đầu train model cho output
giống hệt NuSEE cùng trọng số. Mọi khác biệt sau khi train vì thế đến từ việc head học cách dùng f_1..f_(g-1).
Mixer tuyến tính (không activation): activation sẽ làm hỏng tính chất khởi tạo trên, vì f_g có giá trị âm.
"""
import torch
from torch import nn

from nuse_e.models.nuse_e import NuSEE


class ExitFuseNuSEE(NuSEE):
    def __init__(self, stats, arch, channels=3):
        super().__init__(stats, arch, channels)
        w = self.width
        self.mixers = nn.ModuleList([nn.Identity()])                 # exit 1 chỉ có f_1: không cần fuse
        for g in range(1, len(self.groups)):
            mixer = nn.Conv2d(w * (g + 1), w, 1)
            with torch.no_grad():
                mixer.weight.zero_()
                mixer.bias.zero_()
                mixer.weight[:, g * w:, 0, 0] = torch.eye(w)
            self.mixers.append(mixer)

    def coefficients(self, y, k):
        g = self.budgets.index(k)
        f, feats, parts = self.stem(y), [], []
        for j in range(g + 1):
            f = self.stages[j](f)
            feats.append(f)
            fused = f if j == 0 else self.mixers[j](torch.cat(feats, 1))
            parts.append(self.heads[j](fused, y).reshape(y.shape[0], self.channels, self.groups[j], *y.shape[-2:]))
        return torch.cat(parts, 2).flatten(1, 2)

    def extra_inference_params(self, g):
        """Tham số mixer của exit 1..g+1 (cộng thêm vào công thức tham số của NuSEE)."""
        return sum(p.numel() for m in self.mixers[1:g + 1] for p in m.parameters())

    def params_for(self, k):
        return super().params_for(k) + self.extra_inference_params(self.budgets.index(k))
