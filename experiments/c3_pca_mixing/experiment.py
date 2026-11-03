"""c3: e80 và d8 với basis học được (PCA mixing của Luân), so với basis PCA cố định cùng chi phí suy luận."""
from nuse_e.config import VARIANTS as BASE

IMPL = "pca_mixing.model:MixingNuSEE"
MIXING = {"basis_lr_scale": 0.1}                        # R học với 0.1 × lr chung = 1e-4 → 1e-7 (như run của Luân)
E80, D8 = BASE["e80"], BASE["d8"]                       # e80: 7 block, exit K = 5 / 9 / 15; d8: 8 block, K = 2 / 5 / 9 / 15

VARIANTS = {
    "e80": E80,                                         # tham chiếu: 132,003 tham số ở K = 15
    "d8": D8,                                           # tham chiếu: 150,499
    "e80mix": dict(E80, impl=IMPL, options=MIXING),     # như e80 khi suy luận
    "d8mix": dict(D8, impl=IMPL, options=MIXING),       # như d8 khi suy luận
}
