"""c1: conv ba nhánh trong trunk e80 khi train, gộp thành một Conv3x3 khi suy luận (chi phí suy luận = e80)."""
from nuse_e.config import VARIANTS as BASE

IMPL = "rep_branches.model:RepBranchNuSEE"
E80 = BASE["e80"]                                       # 7 block (5, 1, 1), exit K = 5 / 9 / 15

VARIANTS = {
    "e80": E80,                                         # tham chiếu
    "e80rep": dict(E80, block="rep"),                   # 3x3 + 1x1 + identity (RepVGG; RepConv có sẵn của nuse_e)
    "e80acb": dict(E80, block="acb", impl=IMPL),        # 3x3 + 1x3 + 3x1 (ACNet)
    "e80ecb": dict(E80, block="ecb", impl=IMPL),        # 3x3 + (1x1 -> 3x3) + (1x1 -> Sobel/Laplacian) (ECBSR)
}
