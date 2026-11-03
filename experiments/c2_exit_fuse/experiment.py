"""c2: concat các stage + fuse Conv1x1 ở mỗi exit; tách tác dụng của fuse với tác dụng của block thứ 8."""
from nuse_e.config import VARIANTS as BASE

IMPL = "exit_fuse.model:ExitFuseNuSEE"
E80 = BASE["e80"]                                       # 7 block (5, 1, 1), exit K = 5 / 9 / 15

VARIANTS = {
    "e80": E80,                                         # tham chiếu: 132,003 tham số ở K = 15
    "e80f": dict(E80, impl=IMPL),                       # + fuse: 137,187
    "e80d8": dict(E80, total_blocks=8),                 # chỉ thêm 1 block (ablation chiều sâu): 150,499
    "e80f8": dict(E80, total_blocks=8, impl=IMPL),      # thêm block + fuse: 155,683 (trần 156,856)
}
