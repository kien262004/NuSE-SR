"""Conv3x3 ba nhánh khi train; `fuse()` trả về một nn.Conv2d 3x3 cho output giống hệt (sai số làm tròn).

    ACBConv     3x3 + 1x3 + 3x1                          ACNet (Ding et al., ICCV 2019)
    ECBConv     3x3 + (1x1 -> 3x3) + (1x1 -> lọc cạnh)   ECBSR (Zhang et al., ACM MM 2021), rút còn ba nhánh
    (bản RepVGG 3x3 + 1x1 + identity đã có sẵn: RepConv trong nuse_e/models/blocks.py)

Mọi nhánh đều tuyến tính, không BN, nên tổng của chúng vẫn là một phép conv 3x3. Nhánh có conv 1x1 đứng trước conv 3x3
được pad bằng bias của conv 1x1 thay vì 0 (như ECBSR): khi đó viền ảnh cũng khớp đúng với conv đã gộp, vì conv gộp
nhìn thấy x = 0 ngoài ảnh, và conv 1x1 biến x = 0 thành đúng bias của nó.
"""
import torch
from torch import nn
from torch.nn import functional as F

# Sobel-x, Sobel-y, Laplacian: đạo hàm bậc 1 theo hai hướng và bậc 2 (nhánh cạnh của ECBSR).
EDGE_MASKS = torch.tensor([[[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
                           [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]],
                           [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]])


def conv3x3(weight, bias):
    conv = nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1).to(weight)
    conv.weight.copy_(weight)
    conv.bias.copy_(bias)
    return conv


def pad_with_bias(t, bias):
    """Pad 1 pixel mỗi cạnh, giá trị pad = bias của từng kênh (tương đương conv 1x1 nhìn thấy x = 0 ngoài ảnh)."""
    b = bias.view(1, -1, 1, 1)
    return F.pad(t - b, (1, 1, 1, 1)) + b


class ACBConv(nn.Module):
    """3x3 + 1x3 + 3x1: hai nhánh dải tăng cường hàng giữa và cột giữa của kernel ("bộ xương" của kernel)."""

    def __init__(self, width):
        super().__init__()
        self.square = nn.Conv2d(width, width, 3, padding=1)
        self.hor = nn.Conv2d(width, width, (1, 3), padding=(0, 1))
        self.ver = nn.Conv2d(width, width, (3, 1), padding=(1, 0))

    def forward(self, x):
        return self.square(x) + self.hor(x) + self.ver(x)

    @torch.no_grad()
    def fuse(self):
        weight = self.square.weight.clone()
        weight[:, :, 1:2, :] += self.hor.weight
        weight[:, :, :, 1:2] += self.ver.weight
        return conv3x3(weight, self.square.bias + self.hor.bias + self.ver.bias)


class ECBConv(nn.Module):
    """3x3 + (1x1 nở ra `expand` lần -> 3x3 thu lại) + (1x1 -> depthwise lọc cạnh).

    Nhánh cạnh: mỗi kênh một tổ hợp học được của Sobel-x, Sobel-y, Laplacian (ECBSR có ba nhánh riêng cho ba bộ lọc;
    ở đây gộp chung một conv 1x1 để giữ đúng ba nhánh). Tỉ lệ khởi tạo nhỏ như ECBSR.
    """

    def __init__(self, width, expand=2):
        super().__init__()
        self.square = nn.Conv2d(width, width, 3, padding=1)
        self.expand = nn.Conv2d(width, expand * width, 1)
        self.squeeze = nn.Conv2d(expand * width, width, 3)              # pad bằng tay (pad_with_bias)
        self.reduce = nn.Conv2d(width, width, 1)
        self.edge_scale = nn.Parameter(torch.randn(width, len(EDGE_MASKS)) * 1e-3)
        self.edge_bias = nn.Parameter(torch.zeros(width))
        self.register_buffer("edge_masks", EDGE_MASKS.clone())

    def edge_kernel(self):
        """(width, 1, 3, 3): kernel depthwise của nhánh cạnh."""
        return torch.einsum("cm,mhw->chw", self.edge_scale, self.edge_masks).unsqueeze(1)

    def forward(self, x):
        out = self.square(x)
        out = out + self.squeeze(pad_with_bias(self.expand(x), self.expand.bias))
        edge = pad_with_bias(self.reduce(x), self.reduce.bias)
        return out + F.conv2d(edge, self.edge_kernel(), self.edge_bias, groups=edge.shape[1])

    @torch.no_grad()
    def fuse(self):
        w1, b1 = self.expand.weight[..., 0, 0], self.expand.bias            # (m, c), (m,)
        w2, b2 = self.squeeze.weight, self.squeeze.bias                     # (o, m, 3, 3), (o,)
        weight = self.square.weight + torch.einsum("omhw,mc->ochw", w2, w1)
        bias = self.square.bias + b2 + torch.einsum("omhw,m->o", w2, b1)
        k, w3, b3 = self.edge_kernel()[:, 0], self.reduce.weight[..., 0, 0], self.reduce.bias   # (o,3,3), (o,c), (o,)
        weight = weight + k.unsqueeze(1) * w3[..., None, None]
        bias = bias + self.edge_bias + b3 * k.sum((1, 2))
        return conv3x3(weight, bias)
