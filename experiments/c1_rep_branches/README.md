# Cách 1: conv ba nhánh, gộp lại khi suy luận (`c1_rep_branches`)

**Câu hỏi.** Lúc train, thay mỗi Conv3x3 trong trunk của e80 bằng ba nhánh conv cộng lại. Khi suy luận gộp chúng về
đúng một Conv3x3. Cách này có tăng PSNR mà không tốn thêm chi phí suy luận không?

**Thiết kế.** Stage, exit (K = 5 / 9 / 15), head, decoder PCA và loss giữ nguyên e80. Chỉ conv trong 7 ResBlock
được thay:

```
lúc train:    x ─┬─ nhánh 1 ─┐
                 ├─ nhánh 2 ─┼─ (+) ─ ReLU ─ ...      (mọi nhánh tuyến tính, không BN)
                 └─ nhánh 3 ─┘
lúc suy luận: x ── Conv3x3 (đã gộp) ─ ReLU ─ ...      = e80, nạp được vào NuSEE thường (strict)
```

Trục K không đổi: model sau khi gộp có đúng tham số, FLOPs và latency của e80 ở mọi K.

| Biến thể | Ba nhánh | Nguồn | Tham số suy luận K = 5 / 9 / 15 | Tham số lúc train |
|---|---|---|---|---|
| `e80` | Conv3x3 (tham chiếu) | | 94,021 / 112,913 / 132,003 | 132,003 |
| `e80rep` | 3x3 + 1x1 + identity | RepVGG (CVPR 2021) | như e80 | 146,787 |
| `e80acb` | 3x3 + 1x3 + 3x1 | ACNet (ICCV 2019) | như e80 | 218,915 |
| `e80ecb` | 3x3 + (1x1 → 3x3, nở 2 lần) + (1x1 → Sobel-x / Sobel-y / Laplacian) | ECBSR (ACM MM 2021) | như e80 | 436,643 |

`e80rep` dùng `RepConv` có sẵn của `nuse_e`. Bản RepConv trước đây (`rep7`) được thử trên exit 2/5/9/15 của v2,
chưa thử trên exit của e80, nên ở đây train lại để so cùng điều kiện.

**Đã biết trước.** rep7 +0.03 dB, rep8 −0.01 dB so với bản không rep. Hai lần train cùng seed lệch nhau tới 0.01 dB.
Kỳ vọng thấp: re-param chỉ đổi đường đi của gradient lúc train, không tăng sức biểu diễn của model đã gộp.
ECB là ứng viên đáng thử nhất, vì nhánh Sobel / Laplacian khớp với dạng của vài vector PCA đầu
(mặt dốc theo hai hướng, dạng yên ngựa trong ô 4×4).

**Coi là có ích khi** PSNR ở K = 15 cao hơn e80 quá 0.01 dB, các K nhỏ không tụt, và cột ràng buộc trong
`comparison/constraints.csv` ghi đạt hết. Train ECB chậm hơn e80 vì nhánh 1x1 → 3x3 nở gấp đôi kênh.

**Hai loại LR.** Mọi biến thể train trên cả LR average pool và LR bicubic (xem [README chung](../README.md)).
So mỗi biến thể với e80 cùng loại LR. Output v3 có sẵn run tham chiếu bicubic (v2, e80, d8).

## File

| File | Nội dung |
|---|---|
| `experiment.py` | Biến thể và thiết lập riêng; pipeline `nuse_e` đọc file này qua `--experiment` |
| `rep_branches/branches.py` | `ACBConv`, `ECBConv` và phép gộp (`fuse()`) |
| `rep_branches/model.py` | `RepBranchNuSEE`: NuSEE với block ba nhánh |
| `test_rep_branches.py` | Gộp đúng cả ở viền ảnh; model đã gộp = e80; A(SR) = y; exit nhỏ không đọc phần sau |
| `kaggle_c1_rep_branches.ipynb` | Notebook Kaggle tự chứa (bản chụp code lúc sinh, xem [README chung](../README.md#notebook-kaggle)) |

## Chạy trên máy (từ gốc repo)

```bash
python -m unittest experiments/c1_rep_branches/test_rep_branches.py
python -m nuse_e.pipeline setup --out outputs/c1 --experiment experiments/c1_rep_branches --train-hr <DIV2K_train_HR> --val-hr <DIV2K_valid_HR> --resume-from <output v3>
python -m nuse_e.pipeline stats test screen train curves select evaluate cost report figures --out outputs/c1
```
