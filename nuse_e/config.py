"""Thiết lập thí nghiệm NuSE-E (trước đây là ô cấu hình của notebook).

Sửa giá trị ở đây rồi chạy lại bước cần thiết. Đường dẫn dữ liệu không nằm ở đây: truyền qua
`python -m nuse_e.pipeline setup ...` (xem README.md).
"""
from types import SimpleNamespace

# ---- Thí nghiệm ----
METHODS = ["average_pool", "bicubic"]               # cách tạo LR
BASELINES = ["SPANV2", "DISP", "PLKSR-Tiny"]        # chỉ dùng checkpoint cũ / số đã báo cáo (LR average pool)
OLD_RUNS = "reuse"   # "reuse": baseline LR average_pool lấy từ checkpoint cũ nếu có; "retrain": train lại tất cả
SHOW_OLD_PLUGIN = True                              # thêm dòng "backbone + NuSE (k=8)" của thí nghiệm cũ

# ---- Biến thể NuSE-E. Cùng quy tắc suy kiến trúc từ phổ PCA; chỉ khác các siêu tham số mà quy tắc không quyết
# định: tổng số block, độ rộng, kiểu block, mục tiêu năng lượng, trọng số loss alpha.
# Thứ tự trong dict = thứ tự ưu tiên train. "NuSE-E" (không có hậu tố) là bản v2, luôn có mặt để đối chiếu.
VARIANTS = {
    "d8":     dict(total_blocks=8, width=32, block="plain"),                                   # sâu thêm 1 block
    "rep7":   dict(total_blocks=7, width=32, block="rep"),                                     # RepConv, suy luận = v2
    "w40":    dict(total_blocks=5, width=40, block="plain"),                                   # rộng hơn, nông hơn
    "rep8":   dict(total_blocks=8, width=32, block="rep"),
    "lk7":    dict(total_blocks=7, width=32, block="lk7"),                                     # depthwise 7x7 trong block
    "e80":    dict(total_blocks=7, width=32, block="plain", energy_targets=(0.8, 0.9, 1.0)),   # bỏ exit K=2 (bản chính)
    "top":    dict(total_blocks=7, width=32, block="plain", alpha="top"),                      # loss chỉ ở K lớn nhất
    "w36":    dict(total_blocks=6, width=36, block="plain"),
    "w40rep": dict(total_blocks=5, width=40, block="rep"),
}
PHASE_B_TOP = 2            # xong LR average pool thì train thêm trên LR bicubic cho N biến thể tốt nhất
SELECT_TOL_DB = 0.01       # chọn biến thể: chênh PSNR K lớn nhất dưới ngưỡng này coi là hoà, lấy bản ít tham số hơn
                           # (hai lần train độc lập cùng seed lệch nhau tới 0.01 dB)

# ---- Ràng buộc cứng: kiểm trước khi train (tham số, latency) và sau khi train (đơn điệu theo K, khớp LR)
PARAM_CAP = 156856         # tham số ở K lớn nhất <= DISP + NuSE k=8
LATENCY_REF = "DISP + NuSE k=8 (cũ)"   # trần latency @LR 256² fp32: đo lại từ checkpoint cũ nếu có
LATENCY_CAP_MS = 8.44      # dùng khi không có checkpoint cũ (DISP + NuSE k=8, T4)
CONSISTENCY_TOL = 1e-5     # max|A(SR) - y| cho phép

# ---- Giao thức train (giống các run baseline cũ) ----
EPOCHS, BATCH, REPEAT = 100, 128, 20                # 800 ảnh × 20 / 128 = 125 iteration mỗi epoch
LR, MIN_LR, GRAD_CLIP, SEED = 1e-3, 1e-6, 1.0, 2026
VAL_EVERY, CHECKPOINT_EVERY = 5, 10                 # đánh giá đủ 100 ảnh (chọn best.pt) mỗi 5 epoch
QUICK_VAL_IMAGES, QUICK_VAL_CROP = 10, 480          # mỗi epoch: 10 crop 480×480 để theo dõi tiến độ
AMP = False                                         # phải giống nhau cho mọi model
WORKERS = 4
MAX_HOURS = None                                    # ví dụ 11.5: dừng nhận run mới sau chừng ấy giờ

# ---- Quy tắc kiến trúc NuSE-E (thiết kế v2, mục 4) ----
DESIGN = dict(energy_targets=(0.5, 0.8, 0.9, 1.0), total_blocks=7, r2_min=0.05, rel_tol=1e-6, min_share=0.5,
              width=32, block="plain", alpha="uniform")   # = bản v2 (tham chiếu)
STATS_REPEAT, STATS_FOLDS = 4, 5

# ---- Đánh giá ----
USE_LPIPS, USE_FID = True, True                     # cần gói lpips, torchmetrics, torch-fidelity
BENCHMARK_NAMES = ["Set5", "Set14", "B100", "BSD100", "Urban100", "Manga109"]
SAVE_SR_IMAGES = False

# ---- Latency ----
LATENCY_SIZES = [(128, 128), (256, 256), (339, 510)]
LATENCY_PRECISIONS = ["fp32", "fp16"]
WARMUP, ITERS, ROUNDS = 30, 300, 10
CPU_LATENCY_SIZE, CPU_ITERS = (128, 128), 20
FLOPS_SIZE = (256, 256)

# ---- Checkpoint của thí nghiệm cũ (tìm trong --old-runs) ----
OLD_NAMES = {"SPANV2": "spanv2_x4", "DISP": "disp_x4", "PLKSR-Tiny": "plksr_tiny_x4",
             "SPANV2 + NuSE k=8 (cũ)": "spanv2_nuse_x4_k8", "DISP + NuSE k=8 (cũ)": "disp_nuse_x4_k8",
             "PLKSR-Tiny + NuSE k=8 (cũ)": "plksr_tiny_nuse_x4_k8"}

# Số đã báo cáo của các run cũ (LR average_pool, DIV2K valid 100 ảnh). Dùng khi không có checkpoint cũ.
OLD_REPORTED = {
    'DISP': {'rgb_psnr': 27.520466, 'rgb_ssim': 0.782759, 'lpips': 0.344047, 'lr_max_abs': 0.10915189981460571, 'params': 163792, 'gflops': 21.3868, 'gpu': 'Tesla T4', 'ms_fp32_339x510': 24.042, 'ms_fp16_339x510': 14.673},
    'DISP + NuSE': {'rgb_psnr': 28.111805, 'rgb_ssim': 0.799086, 'lpips': 0.284612, 'lr_max_abs': 2.384185791015625e-07, 'params': 156856, 'gflops': 20.5311, 'gpu': 'Tesla T4', 'ms_fp32_339x510': 24.651, 'ms_fp16_339x510': 15.698},
    'PLKSR-Tiny': {'rgb_psnr': 28.486712, 'rgb_ssim': 0.809117, 'lpips': 0.272896, 'lr_max_abs': 0.0018892884254455566, 'params': 2370544, 'gflops': 310.2694, 'gpu': 'NVIDIA GeForce RTX 3090', 'ms_fp32_339x510': 76.988, 'ms_fp16_339x510': 51.477},
    'PLKSR-Tiny + NuSE': {'rgb_psnr': 28.548999, 'rgb_ssim': 0.811089, 'lpips': 0.267085, 'lr_max_abs': 2.384185791015625e-07, 'params': 2356696, 'gflops': 308.5078, 'gpu': 'NVIDIA GeForce RTX 3090', 'ms_fp32_339x510': 78.418, 'ms_fp16_339x510': 50.007},
    'SPANV2': {'rgb_psnr': 27.989892, 'rgb_ssim': 0.796165, 'lpips': 0.29634, 'lr_max_abs': 0.050102829933166504, 'params': 140944, 'gflops': 18.3627, 'gpu': 'NVIDIA GeForce RTX 3090', 'ms_fp32_339x510': 9.12, 'ms_fp16_339x510': 6.987},
    'SPANV2 + NuSE': {'rgb_psnr': 28.044281, 'rgb_ssim': 0.797652, 'lpips': 0.288856, 'lr_max_abs': 2.384185791015625e-07, 'params': 139000, 'gflops': 18.1613, 'gpu': 'NVIDIA GeForce RTX 3090', 'ms_fp32_339x510': 9.659, 'ms_fp16_339x510': 7.892},
}


def load(smoke=False, only=None):
    """Bản sao các thiết lập trên. `smoke`: chạy thử cực ngắn để kiểm code. `only`: chỉ giữ các biến thể này."""
    c = SimpleNamespace(**{k: v for k, v in globals().items() if k.isupper()})
    c.VARIANTS = dict(VARIANTS)
    if only is not None:
        c.VARIANTS = {k: v for k, v in VARIANTS.items() if k in only}
    if smoke:
        c.EPOCHS, c.BATCH, c.REPEAT, c.VAL_EVERY, c.WORKERS, c.STATS_REPEAT = 2, 2, 1, 1, 0, 1
        c.QUICK_VAL_IMAGES, c.QUICK_VAL_CROP = 2, 64
        c.LATENCY_SIZES, c.WARMUP, c.ITERS, c.ROUNDS, c.CPU_ITERS = [(32, 32)], 1, 2, 1, 2
        c.USE_LPIPS, c.USE_FID = False, False
        if only is None:
            c.VARIANTS = {k: VARIANTS[k] for k in ("d8", "rep7", "e80")}
        c.PHASE_B_TOP = 1
    c.MAX_ITERS = 2 if smoke else None
    c.SMOKE = smoke
    c.NUSE = ["NuSE-E", *[f"NuSE-E/{v}" for v in c.VARIANTS]]   # "NuSE-E" = bản v2; "NuSE-E/<tên>" = biến thể
    c.MODELS = c.NUSE + c.BASELINES
    return c
