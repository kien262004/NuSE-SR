# Cách 2: concat + fuse ở mỗi exit (`c2_exit_fuse`)

**Câu hỏi.** Head của mỗi nhóm hệ số đọc concat các stage đã chạy (fuse bằng Conv1x1, kiểu IMDN / RFB), thay vì chỉ
đọc stage cuối. Cách này có tăng PSNR không? Phần tăng đến từ fuse, hay chỉ từ việc dùng phần tham số còn dư cho một
block nữa?

**Thiết kế.**

```
y → stem → [stage 1] → f1 ───────────────────────────────→ head 1 → c(1..5)     exit K = 5
                └→ [stage 2] → f2 ─ concat(f1, f2) → 1x1 ──→ head 2 → c(6..9)     exit K = 9
                          └→ [stage 3] → f3 ─ concat(f1, f2, f3) → 1x1 → head 3 → c(10..15)   exit K = 15
```

- Trunk giữ nguyên e80: stage sau vẫn chạy tiếp từ `f`, mixer chỉ đưa vào head.
- Exit g chỉ dùng stage 1..g và mixer 1..g, nên chi phí vẫn tăng theo K và exit nhỏ không phụ thuộc phần sau.
  Test T7 của pipeline kiểm cả mixer.
- Mixer khởi tạo để chọn đúng f_g (ma trận đơn vị, còn lại 0), nên lúc bắt đầu train model bằng e80.
  Mọi khác biệt sau khi train vì thế đến từ việc head học cách dùng các stage trước.
- Mixer tuyến tính, không activation, vì activation làm hỏng tính chất khởi tạo trên.
  Hệ quả: mixer + readout 1x1 của head gộp lại vẫn là một phép tuyến tính trên concat.
  Cách 2 vì thế là "đọc đa tầng", không thêm phi tuyến.

| Biến thể | Block / stage | Fuse | Tham số K = 5 / 9 / 15 | Vai trò |
|---|---|---|---|---|
| `e80` | 5, 1, 1 | không | 94,021 / 112,913 / 132,003 | tham chiếu |
| `e80f` | 5, 1, 1 | có | 94,021 / 114,993 / 137,187 | tác dụng riêng của fuse |
| `e80d8` | 6, 1, 1 | không | 112,517 / 131,409 / 150,499 | tác dụng riêng của block thứ 8 |
| `e80f8` | 6, 1, 1 | có | 112,517 / 133,489 / 155,683 | cả hai; trần 156,856 |

Quy tắc suy kiến trúc đặt block thứ 8 vào stage 1. Vì vậy K = 5 của `e80d8` / `e80f8` đắt hơn e80 (112.5k thay vì 94k).
Đây là đánh đổi với ràng buộc "K nhỏ thì model gọn", cần nêu khi so.

**Đã biết trước.** Thêm 1 block cho khoảng +0.09 dB (d8 so với v2). Hai lần train cùng seed lệch nhau tới 0.01 dB.
Latency đang sát trần (v2 8.47 ms so với 8.44 ms trên T4). Hai conv 1x1 và concat ở độ phân giải LR tuy rẻ nhưng
không miễn phí, nên xem cột latency trong `comparison/screen.csv`.

**Đọc kết quả.**

- `e80f − e80`: fuse có ích không.
- `e80d8 − e80`: chiều sâu.
- `e80f8 − e80d8`: fuse có còn ích khi đã thêm block không.

Fuse coi là có ích khi chênh vượt 0.01 dB ở K = 9 và K = 15, và qua sàng latency.

**Hai loại LR.** Mọi biến thể train trên cả LR average pool và LR bicubic (xem [README chung](../README.md)).
So mỗi biến thể với e80 cùng loại LR. Output v3 có sẵn run tham chiếu bicubic (v2, e80, d8).

## File

| File | Nội dung |
|---|---|
| `experiment.py` | Biến thể và thiết lập riêng; pipeline `nuse_e` đọc file này qua `--experiment` |
| `exit_fuse/model.py` | `ExitFuseNuSEE`: NuSEE với mixer Conv1x1 ở mỗi exit |
| `test_exit_fuse.py` | Lúc khởi tạo bằng NuSEE; exit nhỏ không đọc stage / head / mixer sau; tham số đúng công thức và dưới trần |
| `kaggle_c2_exit_fuse.ipynb` | Notebook Kaggle tự chứa (bản chụp code lúc sinh, xem [README chung](../README.md#notebook-kaggle)) |

## Chạy trên máy (từ gốc repo)

```bash
python -m unittest experiments/c2_exit_fuse/test_exit_fuse.py
python -m nuse_e.pipeline setup --out outputs/c2 --experiment experiments/c2_exit_fuse --train-hr <DIV2K_train_HR> --val-hr <DIV2K_valid_HR> --resume-from <output v3>
python -m nuse_e.pipeline stats test screen train curves select evaluate cost report figures --out outputs/c2
```
