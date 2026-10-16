# Experiment configuration

Each experiment YAML has three main sections: `model`, `data`, and `train`.
Optional sections are `pca`, `evaluation`, `runtime`, `logging`, and `metadata`.
The same file can be passed to `fit_pca.py`, `train.py`, and `test.py`;
each command selects its own settings. CLI arguments override YAML defaults.
Legacy flat YAML files are still accepted.

Training and evaluation also accept an optional `logging` section for W&B:

```yaml
logging:
  enabled: true
  project: NuSE-SR
  entity: null
  name: null
  mode: online
  tags: [div2k, x4]
```

Install the optional dependency with `python -m pip install -e '.[wandb]'`, then
authenticate once with `wandb login` or provide `WANDB_API_KEY` through the
server's secret manager. Training sends batch loss and epoch validation metrics,
a train LR/SR/HR preview every 10 batches, and up to four validation SR/HR
pairs after each validation. Evaluation sends summary metrics and a per-image
metrics table.

```yaml
model:
  scale: 4
  rank: 8
  width: 64
  basis: basis/avgpool_x4_rank8.pt
data:
  train_dataset: div2k
  val_dataset: div2k
  train_hr: data/DIV2K_train_HR
  val_hr: data/DIV2K_valid_HR
  resize:
    method: average_pool
    size: [64, 64]
    antialias: true
  val_dataset_options:
    resize:
      size: null
train:
  output: runs/nusesr_x4_k8
  epochs: 100
  batch_size: 8
  lr: 0.001
```

LR images are generated in `DIV2KDataset.__getitem__`, after HR cropping and
augmentation. They are returned as `input`, with the corresponding HR as
`target`; no image export step is needed.

`data.resize.method` selects `average_pool`, `bicubic`, `bilinear`, or a custom
`module:function` accepting `(hr_chw, scale)`. Bicubic and bilinear use PyTorch
interpolation, with `align_corners=False` and configurable antialiasing.

`data.resize.size` is the LR size in `[height, width]` order. In training,
this determines an HR crop of `[height * model.scale, width * model.scale]`.
Rectangular crops use flips and rotations of 0/180 degrees so batch dimensions
stay consistent. With `size: null`, LR dimensions are HR dimensions divided
by `model.scale`, using `data.crop_size` during training and the full mod-cropped
HR image during validation/test. A fixed evaluation size must match the full
HR dimensions divided by the model scale; it never silently resizes the target.

Per-split `train_dataset_options.resize`, `val_dataset_options.resize`, and
`test_dataset_options.resize` override individual fields of `data.resize`.
Global resize settings apply to the built-in `div2k`/`hr_folder` datasets;
custom datasets retain their own constructor options.

`configs/train_div2k_bicubic.yaml` selects bicubic generation in the dataloader.
Model functions remain unchanged. The current NuSESR training/evaluation still
requires `input == AveragePoolSR(target)` and will reject bicubic pairs.
Use the average-pool config for the current coefficient training pipeline.

```bash
python fit_pca.py --config configs/train_div2k.yaml
python train.py --config configs/train_div2k.yaml
python test.py --config configs/train_div2k.yaml
```
