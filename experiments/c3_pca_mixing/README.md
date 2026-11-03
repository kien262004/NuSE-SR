# Cách 3: e80 / d8 với basis học được (PCA mixing của Luân)

Train lại e80 và d8 với decoder có basis học được, so với e80 / d8 dùng basis PCA cố định. Chi phí suy luận giữ nguyên.

## Ý tưởng gốc (Luân, nhánh `exp/pca15-to-8-mixing`)

NuSE-SR cắm vào backbone dự đoán 8 hệ số trên 8 thành phần PCA có phương sai lớn nhất. Luân cho rằng 8 hướng phương sai
lớn nhất chưa chắc là 8 hướng mạng dự đoán tốt nhất từ LR. Vì vậy Luân fit PCA-15 rồi học một basis 8 chiều trực chuẩn
trong span đó: `B = V₁₅ · QR(raw)`. `raw` khởi tạo để B = PCA-8, train bằng MSE ảnh, R dùng lr riêng 1e-4 → 1e-7.

## Áp vào NuSE-E

Với LR average pool, NuSE-E dùng cả 15 chiều null space và các exit lồng nhau: exit K lấy K thành phần PCA đầu.
Ở K = 15 basis đã phủ hết null space, nên mọi phép xoay đều cho cùng kết quả. Việc trộn chỉ có tác dụng ở các exit nhỏ:
e80 ở K = 5, 9; d8 ở K = 2, 5, 9.

| | e80 / d8 | e80mix / d8mix |
|---|---|---|
| decoder | U = Q V (PCA, cố định) | U' = U · R, R = QR(I + 0,1·Δ) trực chuẩn, Δ = 0 lúc đầu |
| exit K dùng | K thành phần PCA đầu | K cột đầu của U': mỗi exit học không gian con K chiều của nó |
| loss (average pool) | MSE hệ số (Định lý 4) | MSE hệ số + năng lượng bị cắt Σ_K α_K Σ_{i>K} c'ᵢ² = Σ_K α_K · MSE ảnh |
| lr của R | — | 1e-4 → 1e-7 (giống run của Luân) |
| suy luận | | `fuse()` gộp R vào decoder: tham số, GFLOPs, latency như e80 / d8 |

Những điểm giữ nguyên:

- **A(SR) = y.** U' vẫn nằm trong null space, vì U = Q V và R chỉ xoay trong span đó.
- **Lúc bắt đầu, model giống hệt e80 / d8 cùng trọng số.** Vì Δ = 0 nên R = I. Mọi khác biệt sau khi train đều đến từ R.
- **Trunk, số block, exit và nhánh Wiener** giống e80 / d8. Hai bên chỉ khác nhau ở decoder.

Hai điểm chi tiết:

- **Năng lượng bị cắt.** Phần này là hằng số khi basis cố định nên NuSEE bỏ đi. Khi basis học được thì nó phụ thuộc R và
  chính nó đẩy R về các hướng dễ dự đoán, nên không được bỏ.
- **Học R mà không sửa trainer chung.** R được học với lr 1e-4 nhờ tham số hoá `I + 0,1·Δ`: với Adam, mỗi bước trên Δ dài
  khoảng lr, nên trên R chỉ còn khoảng 0,1·lr. Muốn đổi thì sửa `MIXING["basis_lr_scale"]` trong `experiment.py`.

## Các biến thể (`experiment.py`)

| Tên | Là gì | Tham số ở K = 15 |
|---|---|---|
| `e80` | tham chiếu (bản chính của bài) | 132,003 |
| `d8` | tham chiếu | 150,499 |
| `e80mix` | e80 + basis học được | 132,003 |
| `d8mix` | d8 + basis học được | 150,499 |

**Hai loại LR.** Mọi biến thể train trên cả LR average pool và LR bicubic (xem [README chung](../README.md)).
So mỗi biến thể với e80 / d8 cùng loại LR. Output v3 có sẵn run tham chiếu bicubic (v2, e80, d8).

Với LR bicubic, NuSEE vốn đã train bằng MSE ảnh qua decoder, nên gradient tự đến R; loss năng lượng bị cắt chỉ cần
cho LR average pool. Ở K lớn nhất basis vẫn phủ hết null space, nên trộn cũng chỉ có tác dụng ở các exit nhỏ.

## Chạy

Kiểm code trước (không cần dữ liệu, chạy từ gốc repo):

```bash
python -m unittest experiments/c3_pca_mixing/test_pca_mixing.py
```

Train và đánh giá (cần GPU). Nếu tiếp tục từ output v3, `e80` và `d8` có `DONE` sẵn ở cả hai loại LR, nên chỉ
train 4 run mới (`e80mix`, `d8mix` × average pool, bicubic):

```bash
python -m nuse_e.pipeline setup --experiment experiments/c3_pca_mixing --resume-from ~/Downloads/notebook_v3/notebook \
    --train-hr /path/DIV2K_train_HR --val-hr /path/DIV2K_valid_HR --old-runs /path/nuse_old_runs --out outputs/c3
python -m nuse_e.pipeline stats test screen train select evaluate cost report figures --out outputs/c3
```

Nếu muốn train lại cả `e80` / `d8` trên cùng máy để đối chứng thì bỏ `--resume-from`. Khi đó bản v2 `NuSE-E` cũng được
train, vì pipeline luôn giữ nó. Chạy thử trên CPU: thêm `--smoke --device cpu`.

Notebook Kaggle tự chứa: `kaggle_c3_pca_mixing.ipynb` (bản chụp code lúc sinh, xem [README chung](../README.md#notebook-kaggle)).

## Đọc kết quả

So `e80mix` với `e80` và `d8mix` với `d8` ở **từng K**, trong `comparison/variants.csv` và `report.md`. Chênh dưới
0,01 dB là nhiễu, vì hai lần train v3 cùng seed đã lệch nhau tới mức đó.

- **Kỳ vọng:** có lợi ở các exit nhỏ. K = 15 chỉ thay đổi gián tiếp, qua trunk và head dùng chung.
- **Nếu K nhỏ tăng mà K = 15 không giảm:** basis học được là cải tiến miễn phí cho chế độ ngân sách thấp.
- **Nếu không tăng:** PCA đã gần tối ưu cho bài này. Đây vẫn là một ablation có ích cho bài báo.
