# NuSE-E: code thí nghiệm (thay cho notebook)

Toàn bộ quy trình của notebook `NuSE-E_train_compare.ipynb` (thống kê PCA, suy kiến trúc, test, sàng, train, đánh giá,
latency, bảng và hình), tách thành các package Python nhỏ, chạy từ dòng lệnh.

## Cấu trúc

Các package xếp theo tầng: tầng dưới không import tầng trên, nên đọc từ trên xuống là đọc từ nền lên.

| Package / file | Nội dung | Đọc khi muốn hiểu |
|---|---|---|
| `core/` | `constants.py` (hệ số phóng 4, đuôi ảnh), `io.py` (seed, ghi JSON / CSV) | |
| `operators.py` | Toán tử đo A (lấy trung bình 4×4, bicubic) và giả nghịch đảo A⁺ | LR được tạo thế nào, bước nhất quán dữ liệu |
| `data/` | `images.py` (đọc ảnh, tập valid), `cache.py` (cache ảnh train uint8 phẳng), `loaders.py` (crop, lật, xoay) | dữ liệu train |
| `nullspace/` | `basis.py` (cơ sở Helmert, ô 4×4, đối xứng D4), `statistics.py` (PCA, Wiener, R² → `stats.pt`), `architecture.py` (quy tắc suy exit, nhóm, số block) | phần lý thuyết, vì sao kiến trúc như vậy |
| `models/` | `blocks.py` (ResBlock, RepConv), `nuse_e.py` (NuSE-E), `baselines.py` (SPANV2, DISP, PLKSR-Tiny), `registry.py` (tạo, nạp checkpoint) | mạng chạy thế nào |
| `evaluation/` | `metrics.py` (PSNR, SSIM), `perceptual.py` (LPIPS, FID), `evaluate.py` (đánh giá trên tập ảnh), `cost.py` (tham số, GFLOPs, latency) | số liệu được đo thế nào |
| `training/` | `trainer.py` (vòng train một run), `run.py` (chạy một run từ `config.json`) | cách train |
| `pipeline/` | Mỗi bước thí nghiệm một module: `prepare`, `checks`, `training`, `selection`, `evaluation`, `report`, `figures`; `context.py` là bối cảnh dùng chung; `__main__.py` là dòng lệnh | thứ tự chạy thí nghiệm |
| `config.py` | Thiết lập: biến thể, ràng buộc cứng, giao thức train, đánh giá, latency | muốn đổi thí nghiệm |

Dùng model trong code khác:

```python
from nuse_e.models import load_run
model, checkpoint = load_run("outputs/nuse_e/runs/average_pool/nuse_e_e80/best.pt")
sr = model(lr, 5)        # exit K = 5; bỏ số K để chạy exit lớn nhất
```

## Khác notebook ở đâu

- **Mỗi bước ghi kết quả ra đĩa và đọc lại từ đĩa.** Chạy lại một bước bất kỳ không cần chạy lại các bước trước, và không còn lỗi kiểu `NameError: METHODS` sau khi khởi động lại kernel.
- **Đã sửa `KeyError: 'width'`** khi chọn biến thể trên checkpoint v2 cũ (checkpoint đó không lưu `width`, `block`, `alpha`).
- **Chọn biến thể có ngưỡng hoà** `SELECT_TOL_DB = 0.01`: chênh PSNR dưới 0,01 dB (bằng độ lệch giữa hai lần train) coi là hoà và lấy bản ít tham số hơn. Trên kết quả v3, quy tắc này chọn e80 thay vì d8 (hơn 0,0008 dB).
- Code tách thành các package nhỏ theo tầng (bảng ở trên). Nội dung từng hàm giữ nguyên so với notebook; chạy thử cùng dữ liệu cho kết quả giống hệt bản một file.
- Không phụ thuộc Kaggle: đường dẫn dữ liệu truyền qua dòng lệnh. Vẫn chạy được trên Kaggle (xem cuối file).
- Bảng tổng hợp ghi ra `comparison/report.md` thay vì hiện trong notebook; hình ghi ra `comparison/figures/`.

## Cài đặt

Chạy từ thư mục gốc repo. Cần Python ≥ 3.10, `torch`, `numpy`, `Pillow`, `matplotlib`. LPIPS và FID cần thêm:

```bash
pip install lpips torchmetrics torch-fidelity
```

Thiếu các gói này thì bước `evaluate` bỏ LPIPS / FID và báo ra màn hình, các metric khác vẫn chạy.

## Chạy

Lần đầu, khai báo dữ liệu. Đường dẫn được nhớ trong `<out>/paths.json`, các bước sau không cần truyền lại:

```bash
python -m nuse_e.pipeline setup --train-hr /path/DIV2K_train_HR --val-hr /path/DIV2K_valid_HR
```

Rồi chạy hết:

```bash
python -m nuse_e.pipeline all
```

Hoặc từng bước, theo thứ tự:

| Bước | Làm gì | Ghi ra |
|---|---|---|
| `setup` | Tìm dữ liệu, tạo cache ảnh, kiểm checkpoint cũ | `paths.json`, `cache/` |
| `stats` | PCA null space (đối xứng D4), Wiener, R²; suy kiến trúc mọi biến thể | `stats/<lr>/stats.pt`, `arch*.json` |
| `test` | Test chấp nhận: UᵀU = I, A(SR) = y với trọng số ngẫu nhiên, co giãn theo K, số tham số, gộp RepConv, Wiener | in ra; dừng nếu có FAIL |
| `screen` | Tham số và latency ở K lớn nhất; vượt trần thì không train | `comparison/screen.json` |
| `train` | Train mọi run còn thiếu, mỗi GPU một run; xong pha A thì pha B trên LR bicubic | `runs/<lr>/<model>/` |
| `curves` | Hình đường học | `figures/1_training_curves.png` |
| `select` | Chọn biến thể tốt nhất mỗi loại LR | `comparison/selected.json`, `variants.csv` |
| `evaluate` | PSNR/SSIM RGB, kênh Y, LPIPS, FID, max\|A(SR) − y\| trên DIV2K valid và benchmark | `comparison/<lr>/report.json`, `quality.csv`, `per_image.csv` |
| `cost` | Tham số, GFLOPs, latency GPU (và CPU) | `comparison/cost.json`, `latency.json` |
| `report` | Bảng tổng hợp, ràng buộc cứng sau train, so hai loại LR | `comparison/report.md`, `constraints.csv` |
| `figures` | PSNR theo chi phí, hai loại LR, phân rã latency, từng ảnh, crop trực quan | `comparison/figures/` |

Có thể ghép nhiều bước: `python -m nuse_e.pipeline evaluate cost report figures`.

### Tuỳ chọn hay dùng

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--out DIR` | Thư mục output, mặc định `outputs/nuse_e` (git bỏ qua `outputs/`) |
| `--data-root DIR` | Tự tìm `DIV2K_train_HR` / `DIV2K_valid_HR` trong thư mục này thay vì truyền từng đường dẫn |
| `--old-runs DIR` | Thư mục chứa checkpoint cũ (`spanv2_x4/best.pt`, `disp_nuse_x4_k8/best.pt`, ...): baseline được đo lại, trần latency đo lại trên cùng GPU |
| `--benchmarks DIR` | Thư mục chứa Set5, Set14, B100, Urban100, Manga109 (ảnh HR) |
| `--resume-from DIR` | Bước `setup`: chép `stats/` và `runs/` của phiên trước; run có file `DONE` không train lại |
| `--variants e80,d8` | Chỉ dùng các biến thể này (bản v2 luôn có) |
| `--gpus 0,1` | Bước `train`: GPU dùng, mỗi GPU một run |
| `--device cpu` | Ép chạy CPU |
| `--smoke` | Chạy thử cực ngắn (2 bước train, ít ảnh) để kiểm code |

Không có GPU thì bước `screen` chỉ kiểm số tham số, không kiểm latency, nên các biến thể rộng (w40, w36, lk7, ...) sẽ không bị loại. Muốn chỉ train một số biến thể thì dùng `--variants`.

### Tiếp tục từ kết quả Kaggle v3

Không train lại gì, chỉ chạy phần đánh giá (cần GPU để có latency):

```bash
python -m nuse_e.pipeline setup --resume-from ~/Downloads/notebook_v3/notebook \
    --train-hr /path/DIV2K_train_HR --val-hr /path/DIV2K_valid_HR --old-runs /path/nuse_old_runs
python -m nuse_e.pipeline stats test screen train select evaluate cost report figures
```

Bước `stats` dùng lại `stats.pt` đã có; bước `train` thấy file `DONE` nên bỏ qua mọi run.

### Chạy lại một run riêng

```bash
python -m nuse_e.training.run outputs/nuse_e/runs/average_pool/nuse_e_e80/config.json
```

Run dở tiếp tục từ `last.pt`. Muốn train lại từ đầu thì xoá thư mục run đó.

## Thư mục output

```
outputs/nuse_e/
  paths.json                     đường dẫn dữ liệu đã khai báo
  cache/                         cache ảnh train (uint8 phẳng) và valid (BMP)
  stats/<lr>/                    stats.pt, arch.json (v2), arch_<biến thể>.json
  runs/<lr>/<model>/             config.json, train.log, history.json, best.pt, last.pt, epoch_XXXX.pt, DONE
  comparison/                    screen, selected, variants, cost, latency, constraints, report.md
  comparison/<lr>/               report.json, quality.csv, per_image.csv, summary.json
  comparison/figures/            hình 1–6
```

`<lr>` là `average_pool` hoặc `bicubic`; `<model>` là `nuse_e` (bản v2) hoặc `nuse_e_<biến thể>`.

## Chạy trên Kaggle

Đưa thư mục `nuse_e/` lên (ví dụ upload thành dataset), rồi trong một ô notebook:

```bash
!cp -r /kaggle/input/<dataset-code>/nuse_e /kaggle/working/ && cd /kaggle/working && \
 python -m nuse_e.pipeline setup --out /kaggle/working/out --data-root /kaggle/input --cache /tmp/nuse_cache \
     --old-runs /kaggle/input --resume-from /kaggle/input/<output-phiên-trước>/notebook
!cd /kaggle/working && python -m nuse_e.pipeline all --out /kaggle/working/out
```

Đặt `MAX_HOURS = 11.5` trong `config.py` để bước `train` dừng nhận run mới trước khi hết phiên 12 giờ.
