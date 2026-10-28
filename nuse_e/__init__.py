"""NuSE-E: siêu phân giải ×4 bằng PCA trong null space, một model nhiều exit theo số hệ số K.

Các package con, xếp từ tầng thấp lên tầng cao (tầng sau chỉ import tầng trước):

    core/         hằng số, seed, ghi JSON / CSV
    operators.py  toán tử đo A (average pool, bicubic) và giả nghịch đảo A⁺
    data/         đọc ảnh, cache ảnh train, bộ lấy crop
    nullspace/    cơ sở Helmert, thống kê PCA và Wiener, quy tắc suy kiến trúc từ phổ PCA
    models/       NuSE-E, các khối của trunk, baseline, tạo và nạp model
    evaluation/   PSNR, SSIM, LPIPS, FID, đánh giá trên tập ảnh, tham số, GFLOPs, latency
    training/     vòng train một run (python -m nuse_e.training.run <config.json>)
    pipeline/     các bước của thí nghiệm (python -m nuse_e.pipeline <bước>)
    config.py     thiết lập thí nghiệm

Xem README.md.
"""
