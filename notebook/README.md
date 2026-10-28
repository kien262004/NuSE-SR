# NuSE-E bản v3: cải thiện trong ràng buộc

File chạy: `NuSE-E_train_compare.ipynb` (một notebook, tự chứa toàn bộ code). Phần mở đầu của notebook có bảng biến thể, ràng buộc và cách chạy; file này tóm tắt lý do.

## Kết quả v2 (phiên Kaggle 28/10) và điều rút ra

LR average pool, DIV2K valid 100 ảnh, cùng giao thức train với các baseline (100 epoch × 125 iteration, batch 128, DIV2K train).

| Model | PSNR | SSIM | LPIPS | tham số | ms fp32 @LR 256² (T4) |
|---|---|---|---|---|---|
| NuSE-E v2 K=2 | 27.40 | 0.7815 | 0.252 | 56,642 | 3.84 |
| NuSE-E v2 K=5 | 27.92 | 0.7947 | 0.275 | 94,021 | 6.05 |
| NuSE-E v2 K=9 | 28.04 | 0.7971 | 0.284 | 112,913 | 7.19 |
| NuSE-E v2 K=15 | 28.07 | 0.7977 | 0.284 | 132,003 | 8.47 |
| SPANV2 / + NuSE k=8 | 27.99 / 28.04 | 0.7962 / 0.7977 | 0.296 / 0.289 | 140,944 / 139,000 | đo trên GPU khác |
| DISP / + NuSE k=8 | 27.52 / 28.11 | 0.7828 / 0.7991 | 0.344 / 0.285 | 163,792 / 156,856 | 8.19 / 8.44 |
| PLKSR-Tiny / + NuSE k=8 | 28.49 / 28.55 | 0.8091 / 0.8111 | 0.273 / 0.267 | 2,370,544 / 2,356,696 | đo trên GPU khác |

LR bicubic: NuSE-E v2 K=3 / 7 / 16 đạt 28.02 / 28.34 / 28.40 dB (nội suy bicubic: 26.83). Chưa có baseline nào trên bicubic.

Những điều đọc ra từ `comparison/average_pool/per_image.csv` và `summary.json`:

1. Phần sai số do cắt bớt K gần bằng 0 từ K=9: sai số nằm ở dự đoán hệ số (91% ở K=9, 100% ở K=15). Giới hạn hiện tại là sức của trunk dùng chung, không phải lý thuyết.
2. Còn dư 25k tham số so với DISP + NuSE k=8 (trần của nhóm lightweight); latency đang bằng nhau.
3. K=2 bị vỡ khối 4×4 (oracle chỉ 29.7 dB): điểm vận hành này không dùng được, dù LPIPS nhìn có vẻ tốt.
4. 150 s/epoch là do nạp dữ liệu (mỗi crop giải mã lại cả ảnh 2K). GPU chỉ cần khoảng 20–30 s/epoch.

## Thay đổi ở v3

| Thay đổi | Tác dụng | Ảnh hưởng đến ràng buộc |
|---|---|---|
| Tập train giữ trong RAM (file uint8 phẳng, crop bằng memmap) | mỗi run 30–60 phút thay vì 4.2 giờ, nên train được 8–10 biến thể một phiên | không đổi giao thức: cùng crop 256, cùng xoay/lật, cùng seed |
| 9 biến thể của trunk (sâu hơn, rộng hơn, RepConv, depthwise 7×7, bỏ K=2, loss chỉ K lớn nhất) | tìm bản dùng hết 157k tham số và trần latency | kiến trúc vẫn do quy tắc suy từ phổ PCA; chỉ đổi N, W, kiểu block, mục tiêu năng lượng, alpha |
| Sàng ràng buộc trước khi train (bước 2b) | biến thể quá tham số hoặc quá chậm không được train | tham số ≤ 156,856; latency @256² fp32 ≤ DISP + NuSE đo cùng GPU |
| Bảng ràng buộc sau khi train (bước 7, `comparison/constraints.csv`) | kiểm đơn điệu theo K và A(SR) = y trên dữ liệu thật | cảnh báo nếu bản được chọn vi phạm |
| Pha B: top-2 biến thể của average pool train thêm trên bicubic | có số bicubic cho bản tốt nhất | |
| Test T11: gộp RepConv cho output giống hệt | RepConv không đổi chi phí suy luận | |

Những gì không đổi: lý thuyết PCA null space (Định lý 1–7), bộ giải mã cố định μ + U_K c, khớp dữ liệu chính xác, head khởi tạo Wiener, loss theo Định lý 4/7, giao thức train chung với baseline.

## Chạy

1. Upload thư mục `notebook/` của phiên trước (output Kaggle) thành dataset, Add Input. Bản v2 không phải train lại.
2. Add Input `nuse_old_runs_kaggle.zip` (checkpoint 6 run cũ) để đo trần latency và baseline trên cùng GPU.
3. Add Input DIV2K; nên thêm Set5/Set14/B100/Urban100/Manga109 (thư mục HR).
4. GPU T4 ×2, Internet bật, Save & Run All. Xem tab Logs: dòng `chờ dữ liệu Xs` mỗi epoch cho biết loader có còn là nút thắt không.

## Đọc kết quả

- `comparison/screen.csv`: biến thể nào qua sàng, tham số và latency ở K lớn nhất.
- `comparison/variants.csv`: PSNR DIV2K valid từng K của mọi biến thể đã train, bản nào được chọn (★).
- `comparison/constraints.csv`: 6 ràng buộc cứng cho từng biến thể.
- `comparison/<lr>/quality.csv`, `per_image.csv`, `figures/`: như phiên trước; hình 2 vẽ mỗi biến thể một đường theo K.

Model được coi là tốt hơn v2 khi: PSNR ở K lớn nhất cao hơn 28.07 (average pool) với tham số ≤ 157k và latency ≤ 8.44 ms trên T4, thắng trên đa số ảnh so với DISP + NuSE k=8 (`5_per_image_average_pool.png`), và cột ràng buộc ghi "đạt hết".
