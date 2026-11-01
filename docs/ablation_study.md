# Local/global backbone ablation

The ablation backbones live in `nullspaceSR/models/ablation/`. They predict PCA
coefficients and use the existing fixed average-pool null-space decoder. LR
generation (`average_pool` or `bicubic`) remains independent of that operator.

| Config name | Predictor | Purpose |
|---|---|---|
| `linear` | grouped Conv7 → all coefficients | linear baseline |
| `cnn` | two Conv3 + ReLU → all coefficients | nonlinear-only baseline |
| `g0_local` | Conv7 + CNN correction to first 2 coefficients/color | local hybrid |
| `g1_global` | G0 with image-average channel gate | global hypothesis |
| `g2_wider` | G0 with width 17 | local capacity control |
| `g3_context` | G0 with dilated second Conv3 | wider finite context control |
| `local_c5` | G0 with correction to first 5 coefficients/color | group ablation |

`g1_global` uses `F * (1 + gamma * tanh(MLP(GAP(F))))`, with `gamma=0` initially.
It has image-wide conditioning but does not perform token-to-token attention.
At RGB K=8 with default widths, G1 has 4,351 parameters and G2 has 4,402
(1.2% more). G2/G3 are **controls to measure**, not guaranteed matched in
latency; matching can change with other settings.
Inspect `reports/runs.csv` on the target hardware and tune widths if needed.

Conv7 trains from random initialization using the shared coefficient loss;
there is no fitted ridge initialization in this study. Each correction head
starts at zero. Set `model.linear_groups: rgb` for dense RGB mixing instead of
the default independent color groups. Add variants with distinct names to
compare correction sizes or widths. `corrected` must not exceed any rank used.

## Run on the server

```bash
python -m pip install -e ".[research]"
# Edit paths in configs/ablation_study.yaml first.
bash scripts/ablation.sh --config configs/ablation_study.yaml
```

The config defaults to `runs/architecture_research/basis/full.pt` from the prior
research run. The basis must be RGB, full rank 15 for x4, fit on **train HR**.
If absent, fit it once and set `basis` in the config:

```bash
python fit_pca.py --config configs/pca_div2k.yaml --rank 15 --output basis/avgpool_x4_rank15.pt
```

With 2 LR methods, one rank, one seed and 7 variants, the default config starts
14 runs of 100 epochs. It is a screening configuration, not a 3-seed conclusion.
After shortlisting, use a **new output directory** and set
`seeds: [2026, 2027, 2028]` and desired ranks, e.g. `[8, 9, 12]`.

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ablation.sh \
  --config configs/ablation_study.yaml > logs/ablation.log 2>&1 &
tail -f logs/ablation.log
```

Use `--device`, `--train-hr`, `--val-hr`, `--basis`, `--output` to override paths
or device. For a tiny execution check, use RGB folders with images at least 32×32:

```bash
bash scripts/ablation.sh --smoke --device cpu --train-hr /path/train \
  --val-hr /path/val --basis /path/full_rank_basis.pt --output runs/ablation_smoke
```

Smoke runs 2 epochs for local/global/context on each configured LR method; use
`--smoke` again for a later `--stage report`. Never use smoke results as evidence.

Completed runs are skipped and interrupted runs resume from epoch-boundary
`last.pt` with optimizer, scheduler and RNG state. Reuse the same command after
interruption. `protocol.json` freezes config, source hashes, PCA hash, image paths
and file metadata, hardware and torch version; a change requires a fresh output
folder. Run only one process per output directory. `--stage report` regenerates
tables from completed runs with the same config and protocol.

## Outputs and interpretation

Each run saves `best.pt`, `last.pt`, `history.json`, `quality.json`,
`per_image.csv`, `components.csv`, `tiles.csv` and `benchmark.json`. Checkpoints
are self-contained and can be evaluated with the regular `test.py --checkpoint`.
Select best epoch using validation RGB PSNR; quality uses raw floats and shave 0
by default. Set `evaluation.lpips: true` and install `.[metrics]` when LPIPS is
needed; its weights must be available on the server.

`reports/runs.csv` reports individual seed runs; `summary.csv` has seed mean and
population std; `paired_vs_local.csv` compares each variant to `g0_local` on
the same images and matched seeds. Its paired image bootstrap is conditional on
the trained seeds and does **not** capture training-seed uncertainty. Use at
least three seeds before drawing conclusions and an independent test set after
selecting the variant. Component MSE/R² use a fixed uniform sample per image;
image PSNR/SSIM use complete validation images.

`tiles.csv` compares whole-image inference against LR tiles with a three-pixel
halo, for the first configured number of validation images. The halo covers the
local 7×7 and dilated paths, while a global gate is recomputed per tile. The
metric exposes full-vs-tiled differences; it is not a tiled latency benchmark.
`benchmark.json` reports batch-one fp32 end-to-end and separated predictor/
decoder timings at the configured LR size, plus MAC estimates and CUDA memory.
Separate stage medians need not add up to end-to-end latency.
