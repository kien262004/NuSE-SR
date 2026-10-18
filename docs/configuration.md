# Experiment configuration

Each experiment YAML has three main sections: `model`, `data`, and `train`.
Optional sections are `pca`, `evaluation`, `runtime`, `logging`, and `metadata`.
The same file can be passed to `fit_pca.py`, `train.py`, and `test.py`;
each command selects its own settings. CLI arguments override YAML defaults.
Legacy flat YAML files are still accepted.

Validation and test report RGB PSNR (higher is better), RGB SSIM (higher), and
LPIPS (lower). Install perceptual metric dependencies before running:

```bash
python -m pip install -e ".[metrics]"
```

```yaml
evaluation:
  lpips: true
  shave: 0
```

LPIPS is enabled by default in both CLIs and initialized once per run. The first
run downloads pretrained AlexNet weights into the PyTorch cache. The metric uses
[official LPIPS](https://github.com/richzhang/PerceptualSimilarity) v0.1 with
AlexNet, frozen weights and evaluation mode. It clips each image to `[0,1]`,
then maps to `[-1,1]`; PSNR/SSIM retain raw float outputs without clipping or
quantization. All three apply the same `shave` and average per-image scores.
LPIPS does not resize images; each side after shaving must be at least 32 pixels.
Grayscale inputs are replicated to three channels for LPIPS.

Use `--no-lpips` or `evaluation.lpips: false` for tiny fixtures or runs without
perceptual metric dependencies. Missing dependencies/weights cause an explicit
error when enabled; the metric is never silently omitted. LPIPS is written to
JSON summaries, per-image CSV files, console output and W&B, alongside PSNR and
SSIM. `best.pt` selection remains based on validation PSNR.

For direct Python evaluation, create `LPIPSMetric().to(device)` once and pass
it as `evaluate(..., lpips_metric=metric)`. Omitting this argument keeps the
dependency-free PSNR/SSIM path.

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
a train LR/SR/HR preview every batch, and up to four validation SR/HR
pairs after each validation. Evaluation sends summary metrics and a per-image
metrics table.

```yaml
model:
  null_space: true
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

`model.null_space` selects the output domain:

- `true`: predict LR coefficient maps, decode using frozen PCA, and add the
  repeated LR base. Requires `model.basis`. Training uses
  coefficient SSE normalized per HR pixel, as before.
- `false`: use IMDN's Conv + PixelShuffle image head and train against HR with
  image MSE. No PCA file is loaded; bicubic, bilinear, and custom pairs are allowed.

When omitted, supplying `model.basis` enables PCA for compatibility with older
configs; otherwise image-space IMDN is used. `--null-space` / `--no-null-space`
override YAML. `model.channels` defaults to 3 in image mode; PCA mode derives
channels from the basis. `rank` is only used in PCA mode.

`IMDN(channels=3, scale=4)` directly returns an HR image.
`IMDN(channels=3, rank=8)` retains the LR coefficient head.
`NuSESR(predictor, decoder=None)` wraps image-space prediction; passing a
`BlockNullSpacePCA` decoder installs the null-space plugin.

`configs/train_div2k_bicubic.yaml` is a complete image-space preset. Run train
and test directly, without fitting PCA. Checkpoints record the selected mode,
and evaluation automatically restores it without an external basis file.
Oracle, coefficient, and null-space consistency metrics are reported only in
PCA mode; both modes report image quality and output-range metrics.

LR generation is independent of the model: `data.resize.method` may be bicubic,
bilinear, or average pooling in either mode. The PCA model always uses its fixed
block-average operator A, regardless of that setting. It enforces `A(SR) == LR`;
training/evaluation do not require `A(HR) == LR`. PCA fitting uses HR only.
`configs/train_div2k_nullspace_bicubic.yaml` trains the fixed average-pool PCA
model using bicubic LR inputs.

If `A(HR) != LR`, exact HR reconstruction is outside the model's constraint.
Coefficient loss still has the same gradient as image MSE because the extra
error is orthogonal to the decoder's basis. Evaluation reports
`measurement_mismatch_mse = MSE(A(HR), LR)` and `pca_truncation_mse` separately.
The legacy `truncation_mse` field is the full oracle residual, including both;
`full_image_mse = truncation_mse + coefficient_image_mse` remains valid.
`dc_max_abs` measures `A(SR) - LR`, using the fixed model operator.

```bash
python fit_pca.py --config configs/train_div2k.yaml
python train.py --config configs/train_div2k.yaml
python test.py --config configs/train_div2k.yaml
```
