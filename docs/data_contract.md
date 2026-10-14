# Giao diện dữ liệu chung

Dataset trả dictionary cho mỗi sample:

```python
{
    "input": lr,          # CPU floating tensor [C,H,W]
    "target": hr,         # CPU floating tensor [C,sH,sW], hoặc None
    "id": "sample_001",   # ID nguồn dữ liệu ổn định
    "meta": {},           # Đường dẫn, degradation, range, preprocessing...
    "extras": {},         # Mask hoặc dữ liệu riêng của dataset
}
```

Dùng `make_sample(lr, hr, sample_id="sample_001", meta=..., extras=...)` để tạo sample. Khi tự trả dictionary, `input/id` bắt buộc; các trường còn lại có thể bỏ nếu không dùng.

## Đầu ra loader

| Trường | Kiểu / kích thước |
|---|---|
| `input` | Tensor `[B,C,H,W]` |
| `target` | Tensor `[B,C,sH,sW]` hoặc `None` |
| `id` | List B chuỗi |
| `meta` | List B dictionaries, giữ metadata riêng từng sample |
| `extras` | List B dictionaries, không tự stack payload |

Tensor trong một batch phải cùng shape/dtype ở mỗi trường. Ảnh khác kích thước dùng batch size 1 hoặc crop/bucket trong dataset. Loader không tự pad/resize vì có thể thay đổi phép đo/loss. Không trộn sample có và không có target trong cùng batch.

## Chọn dataset từ cấu hình

```yaml
train_dataset: my_package.datasets:MyDataset
train_dataset_options:
  manifest_path: data/my_train.json
  split: train
val_dataset: my_package.datasets:MyDataset
val_dataset_options:
  manifest_path: data/my_validation.json
  split: validation
```

`*_dataset_options` là kwargs constructor custom, được truyền nguyên trạng. Constructor trả `torch.utils.data.Dataset` có `__len__` và `__getitem__`. Các tham số crop/repeat/HR folder của CLI chỉ là mặc định cho `div2k`/`hr_folder`; custom dataset tự quy định preprocessing qua options. Scale trong options phải khớp model.

PCA fitting dùng `train_dataset`/`train_dataset_options`; test dùng `test_dataset`/`test_dataset_options`. Với module riêng trong working directory, chạy `python train.py --config ...`; package custom đã cài dùng được với mọi entry point. Có thể đăng ký tên bằng `register_dataset("my_data", MyDataset)` trong chương trình Python; đường dẫn `module:Class` thuận tiện hơn cho CLI vì không cần sửa code train.

Cấu hình DIV2K cũ vẫn chạy. `hr_folder` dùng cùng implementation để đọc thư mục HR của dataset khác.

## Adapter cho dataset cũ

```python
from nullspaceSR.data import MappingDatasetAdapter, make_loader

dataset = MappingDatasetAdapter(existing_dataset, input_key="lr", target_key="hr", id_key="name")
loader = make_loader(dataset, batch_size=8, shuffle=True)
```

Adapter giữ các trường còn lại trong extras. Có thể bọc dataset cũ trong một constructor custom rồi chọn qua cấu hình.

## Engine dùng batch chung

```python
from nullspaceSR.engine import train_one_epoch
from nullspaceSR.evaluation import evaluate

train_metrics = train_one_epoch(model, train_loader, optimizer, device)
validation_metrics, rows = evaluate(model, val_loader, device)
```

Engine đọc input/target và chuyển sang FP32 trên device. Số kênh lấy từ PCA decoder. Evaluator hiện yêu cầu batch size 1; PNG hỗ trợ 1, 3 hoặc 4 kênh. Metric dùng range 1.

Giao diện dùng được cho nhiều nguồn dữ liệu; NuSESR hiện cần cặp `input=A(target)` với average pooling và đúng scale. Engine kiểm tra điều kiện này để tránh dùng coefficient loss trên degradation khác. `target=None` dùng cho inference `model(batch["input"])`, nhưng supervised train, oracle evaluation và PCA fitting cần target.

## Hook tùy chọn

- `set_epoch(epoch)`: điều khiển sampling/augmentation theo epoch; không có vẫn chạy.
- `manifest()`: trả dictionary JSON-compatible để tracking/resume. `sources` có thể chứa list đường dẫn nguồn để kiểm tra train/validation overlap.

Framework không phụ thuộc các thuộc tính `files/root/training`. Nếu không có manifest hook, chỉ ghi tên lớp và độ dài; không thể phát hiện thay đổi nội dung hoặc overlap chỉ từ thông tin này. Dataset mới cần hook này nếu muốn provenance đầy đủ. Không trả tensor/object tùy biến trong manifest.
