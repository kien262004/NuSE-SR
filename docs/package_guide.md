# nullspaceSR

Khung PyTorch cho SR bằng **hệ số PCA trong null space**, với average pooling theo block không chồng lấn. Mạng hoạt động ở miền LR và xuất `C*k` coefficient maps. Decoder cố định sinh chi tiết có trung bình bằng 0 trên mỗi block; inference không chiếu ảnh do mạng sinh.

## Cấu trúc

```text
./
├── configs/                  # Cấu hình PCA/train/test
├── examples/                 # Synthetic demo và một bước train
├── tests/                    # Kiểm tra các bảo đảm toán học
├── src/nullspaceSR/
│   ├── models/               # Coefficient CNN và NuSESR
│   ├── operators/            # Average-pool A và A†
│   ├── patches.py            # Pixel-unshuffle/shuffle theo block
│   ├── pca.py                # Q, streaming PCA, encode/decode, save/load
│   └── losses.py             # Coefficient SSE / số pixel HR
└── README.md                 # Hướng dẫn ở thư mục gốc
```

## Công thức

Với `q=s*s`, dựng cơ sở Helmert `Q [q,q-1]` sao cho `ones.T @ Q=0`, `Q.T @ Q=I`. Fit PCA trên `Q.T @ HR_block` từ tập train, thu mean `m` và components `V_k`. Theo từng block/kênh:

```text
c = predictor(y)          # [B,C*k,H_LR,W_LR]
z = Q @ (m + V_k @ c)     # null detail: A(z)=0
x_hat = repeat_blocks(y) + z
```

`A(x_hat)=y` tới sai số số học. Q, mean và PCA là buffers, không được optimizer cập nhật. Đây là PCA local dùng chung giữa các vị trí, không phải PCA toàn ảnh. Bảo đảm áp dụng cho average pooling đã chọn, không tự áp dụng cho bicubic hay blur. Mô hình không clamp đầu ra vì clamp có thể phá consistency.

Decoder tính trước `U=Q@V_k` và `mu=Q@m` khi khởi tạo, nên inference chỉ tổng hợp `mu + U@c`, không cần tạo tensor trung gian 15 chiều.

## Cài đặt và chạy ví dụ

Tại thư mục gốc project:

```bash
pip install -e .
python examples/smoke_sr.py
python -m unittest discover -s tests
```

## Xây mô hình của bạn

```python
from nullspaceSR import BlockNullSpacePCA, NuSESR, SmallCoefficientCNN

# Loader chỉ yield tensor HR float [B,C,H,W], từ train split.
pca = BlockNullSpacePCA.fit(train_hr_loader, scale=4, rank=8)
pca.save("basis.pt")
model = NuSESR(SmallCoefficientCNN(channels=3, rank=8, width=32), pca)
sr = model(lr)
null_detail = model.predict_null(lr)
```

Thay predictor bằng backbone của bạn, giữ output `[B,C*k,H_LR,W_LR]`. Fit PCA một lần offline rồi dùng `BlockNullSpacePCA.load("basis.pt")`. Di chuyển cả model và inputs về cùng device/dtype. Augment HR trước, sau đó tạo LR và encode nhãn; không chỉ flip các coefficient maps đã encode.

`RNDModel` là alias cho `NuSESR`; constructor mới nhận `(predictor, decoder)`, thay cho `(backbone, operator)` cũ.

## Huấn luyện và phân tích lỗi

```python
from nullspaceSR import coefficient_image_mse

lr = model.operator(hr)
target = model.decoder.encode_hr(hr)  # Oracle label: chỉ train/eval
predicted = model.predict_coefficients(lr)
loss = coefficient_image_mse(predicted, target, channels=3, scale=4)
loss.backward()
```

Với basis trực chuẩn và `lr=A(hr)`, `image_MSE = truncation_image_MSE + coefficient_SSE / HR_pixel_count`. Nếu dùng coefficient MSE thông thường thì hệ số quy đổi là `k/s²`. Rank tối đa là `s²-1`; full rank biểu diễn toàn bộ local null space kể cả khi covariance thiếu hạng. Rank nhỏ có truncation error mà predictor không thể bù.

## Bố cục mở rộng gợi ý

Dataset dùng giao diện chung `input/target/id/meta/extras`. Loader, engine và evaluator không phụ thuộc DIV2K. Có thể thay dataset qua `module:Class` và options trong YAML; xem [hướng dẫn giao diện dữ liệu](data_contract.md).

Khung đã có dataloader DIV2K, train/resume, evaluation và RGB PSNR/SSIM/MSE. Từ thư mục gốc repository:

```bash
python fit_pca.py --config configs/pca_div2k.yaml
python train.py --config configs/train_div2k.yaml
python test.py --config configs/test_div2k.yaml
```

Chuẩn bị `data/DIV2K_train_HR` và `data/DIV2K_valid_HR`, hoặc chỉnh đường dẫn YAML. Mọi đường dẫn tương đối tính từ working directory. LR được sinh từ HR bằng average pooling; không dùng LR bicubic có sẵn. Xem [README gốc](../README.md#pipeline-div2k) để biết giao thức dữ liệu, metric, checkpoint và resume.

Các file bổ sung: `data/div2k.py`, `fit_pca.py`, `train.py`, `test.py`, `evaluation.py`, `metrics.py`, `utils.py`. Demo synthetic không phải kết quả nghiên cứu DIV2K.
