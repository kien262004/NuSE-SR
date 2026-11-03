# Thí nghiệm nhỏ trên NuSE-E

Ba mini project trên model chính e80 (7 block, exit K = 5 / 9 / 15, 132k tham số):

- c1 và c2 thử ý "ba nhánh conv → cộng / concat → fuse → suy luận" trong trunk.
- c3 thử basis học được ở decoder (PCA mixing của Luân). c3 chạy trên cả e80 và d8 (8 block, exit K = 2 / 5 / 9 / 15,
  151k tham số).

Mỗi project có code, test và notebook Kaggle riêng. Cả ba dùng chung pipeline `nuse_e/` (dữ liệu, PCA, train, đánh
giá, ràng buộc), nên số liệu so được trực tiếp với kết quả v3.

| Project | Cách | Đổi gì so với e80 | Chi phí suy luận |
|---|---|---|---|
| [`c1_rep_branches`](c1_rep_branches/README.md) | Cộng ba nhánh, gộp thành một Conv3x3 khi suy luận (RepVGG / ACNet / ECBSR) | conv trong trunk lúc train | bằng e80 ở mọi K |
| [`c2_exit_fuse`](c2_exit_fuse/README.md) | Concat các stage, fuse bằng Conv1x1 ở mỗi exit (IMDN / RFB) | thêm mixer ở exit K = 9, 15 (+ ablation 8 block) | +5.2k tham số ở K = 15 |
| [`c3_pca_mixing`](c3_pca_mixing/README.md) | Basis học được trong null space: exit K dùng K cột đầu của U·R, R trực chuẩn học được (PCA mixing của Luân) | decoder lúc train, trên cả e80 và d8 | bằng e80 / d8 ở mọi K |

## Mini project gắn vào `nuse_e` thế nào

Mỗi project là một thư mục có `experiment.py`. Pipeline nhận thư mục này qua `--experiment` và nhớ nó trong
`<out>/paths.json`, nên chỉ cần truyền một lần ở bước `setup`.

- `experiment.py` định nghĩa các tên viết hoa (thường chỉ `VARIANTS`, kèm hằng riêng như `IMPL`). Các tên này thay cho
  giá trị trong `nuse_e/config.py`. Biến thể của project thay cho biến thể mặc định; bản v2 (`NuSE-E`) vẫn luôn có để
  đối chiếu.
- Mặc định của mọi thí nghiệm: train trên hai loại LR (`METHODS` của `nuse_e/config.py`). Pha A train mọi biến thể trên
  LR average pool, pha B train **mọi** biến thể trên LR bicubic (`PHASE_B_TOP = len(VARIANTS)`, kể cả khi `--smoke`).
  Muốn khác thì đặt `METHODS` / `PHASE_B_TOP` trong `experiment.py`. So mỗi biến thể với tham chiếu cùng loại LR.
- Biến thể cần model riêng thì khai `impl="<module>:<Class>"`, với Class là lớp con của `NuSEE` nằm trong thư mục
  project. `impl` và `options` được lưu vào `arch_<biến thể>.json` và checkpoint, nên `load_run` nạp lại được.
- Lớp con thêm tham số ngoài công thức của NuSEE thì khai `extra_inference_params(g)` (test T8). Module riêng của
  từng exit đặt trong `self.mixers` để test T7 kiểm cả chúng.
- Test của project (`c*/test_*.py`, chạy không cần dữ liệu) dùng chung `testing.py`: thống kê PCA giả, arch dạng e80,
  nạp `experiment.py`, gán trọng số ngẫu nhiên.

Thay đổi trong `nuse_e/` để có các điểm gắn này:

| File | Thay đổi |
|---|---|
| `config.py` | `load(..., overrides)`; thí nghiệm mặc định `PHASE_B_TOP = len(VARIANTS)` |
| `pipeline/context.py` | `--experiment`, `load_experiment` |
| `pipeline/__main__.py` | thêm tuỳ chọn `--experiment` |
| `pipeline/training.py` | thư mục project vào `PYTHONPATH` của tiến trình train |
| `models/registry.py` | `make_nuse` |
| `nullspace/architecture.py` | lưu `impl` / `options` |
| `pipeline/checks.py` | test dùng `make_nuse`; T7 có mixer; T8 có `extra_inference_params`; T11 chạy cho mọi block gộp nhánh |

Không có `--experiment` thì pipeline chạy y như trước.

```bash
for t in experiments/c*/test_*.py; do python -m unittest "$t"; done     # test của cả ba project, từ gốc repo
```

## Notebook Kaggle

Mỗi project có `kaggle_<project>.ipynb` tự chứa: notebook nhúng toàn bộ `nuse_e/` và thư mục project **tại lúc sinh**
(1–2/11/2026), nên trên Kaggle chỉ cần Add Input dữ liệu và output v3 (hướng dẫn ở đầu mỗi notebook). Script sinh
notebook đã bỏ: notebook là bản chụp, sửa code local không đổi notebook, và code nhúng có thể khác code local (ví dụ
test chưa dùng `testing.py`, `experiment.py` còn ghi rõ `METHODS` / `PHASE_B_TOP`). Kết quả không đổi, vì model và
pipeline tương đương.

Mỗi notebook:

1. Chạy unit test của project.
2. Chạy `stats test screen train curves select evaluate cost report figures`.
3. In bảng PSNR từng K kèm chênh so với e80 cùng loại LR.

Chạy bằng Save & Run All (commit) để tắt trình duyệt / máy không ảnh hưởng. Kaggle cắt mỗi lần chạy ở 12 giờ, nên
train dừng sau `TRAIN_HOURS` (mặc định 10.5): run đang train lưu `last.pt` rồi dừng, notebook bỏ qua bước đánh giá và
kết thúc bình thường. Khi đó thêm output của version vừa chạy vào Input rồi Save & Run All lần nữa: notebook gộp mọi
output cũ trong Input (lấy bản tiến xa nhất của từng run) và train tiếp từ epoch đã dừng, kết quả như chạy một mạch.
