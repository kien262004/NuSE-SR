# Server architecture research

This runner implements the first actionable experiments in the architecture
research plan: oracle-vs-rank, component energy, ridge-vs-context and nonlinear
CNN probes. An optional sweep trains independent NuSE/IMDN models per rank and
seed. The model operator remains average pooling for every LR generation method.

## Run on Linux

From the repository root, in the Python environment used for training:

```bash
python -m pip install -e ".[research]"

# Edit train_hr/val_hr, GPU, contexts and budgets in this config first.
python -u -m nullspaceSR.analysis.research --config configs/research_architecture.yaml --stage all
```

The shell wrapper forwards every argument and changes to the repository root:

```bash
bash scripts/research.sh --config configs/research_architecture.yaml --stage all
```

Run in the background on one selected GPU:

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/research.sh \
  --config configs/research_architecture.yaml --stage all \
  > logs/architecture_research.log 2>&1 &
tail -f logs/architecture_research.log
```

Run each stage separately with the **same config/output**:

```bash
bash scripts/research.sh --stage basis
bash scripts/research.sh --stage oracle
bash scripts/research.sh --stage ridge
bash scripts/research.sh --stage cnn
bash scripts/research.sh --stage report
```

`all` runs basis → oracle → ridge → CNN → report, sequentially. It deliberately
excludes the expensive backbone sweep. The default sweep has 72 independent
training runs (2 LR methods × 3 ranks × 3 seeds × 4 variants); shortlist variants
and seeds before launching it:

```bash
bash scripts/research.sh --stage sweep
bash scripts/research.sh --stage report
```

The existing training CLI performs sweep training and validation; each run has
its own YAML, console log, `best.pt` and `last.pt`. RepConv is fused on the loaded
inference model for latency measurements; training checkpoints remain unfused.
The runner does not change existing train configs or launch concurrent jobs.

## Short execution check

Use real split folders containing RGB images at least 32×32 for the default x4
smoke run. This uses two training batches, a single validation image, contexts
1/3, a tiny CNN and two epochs. It is a code check, not a scientific result.

```bash
bash scripts/research.sh --smoke --device cpu \
  --train-hr /path/to/train --val-hr /path/to/val \
  --output runs/research_smoke --stage all
```

Use a different output directory for full experiments. Always include `--smoke`
again when continuing a smoke run, including `--stage report` or `--stage sweep`.

## Protocol and resume

- PCA fits **train HR only**, at rank `scale²-1` (15 for x4). An existing rank-8
  basis is insufficient. Set `pca.basis` to a matching full-rank basis or leave
  it null. Every compared rank slices this one fitted basis and mean.
- `data.methods` changes LR generation only. Quality measurements remain about
  the fixed average-pool model. Bicubic measurement mismatch is reported as a
  separate irreducible error.
- `protocol.json` records the effective config, image paths/sizes/modification
  times, source hashes, external basis hash, device/hardware, torch version and
  CPU thread count. Changing them requires a fresh
  output directory. This prevents silently mixing completed runs from different
  experiments. It does not hash all image contents.
- Completed `DONE` runs are skipped. Interrupted CNN training resumes the last
  epoch with optimizer/scheduler/RNG state. Interrupted ridge fitting restarts
  that context; its normal equations are not checkpointed mid-fit.
- Use the same command to continue after interruption. A crash during evaluation
  recomputes that run's evaluation. `report` can regenerate tables/figures from
  completed evaluation files without fitting PCA or initializing LPIPS.
- Run only one process per output directory. For separate experiments/GPUs use
  separate outputs and avoid comparing changed data/protocols as matched runs.

## What is measured

Ridge takes replicate-padded LR windows of size r×r. `same_channel` fits each
channel's K coefficients from that color channel; `rgb` fits all 3K outputs
from all colors. Float64 XᵀX/XᵀY are accumulated batch by batch. Only sampled
windows are gathered; the full training patch matrix is never materialized.
Ridge minimizes mean squared residual plus `alpha * ||W||²`; the intercept is
not regularized. The default fixed alpha is not claimed optimal for all contexts.

CNN probes use 3×3 convolutions + ReLU and a 1×1 output head, with exactly the
requested theoretical receptive field (r=1 uses 1×1 layers). They predict full
rank, use RGB jointly, and train for a fixed number of epochs. Evaluation uses
the final epoch, avoiding a different validation-based early-stop rule for each
probe. These probes diagnose context/nonlinearity; their differing parameter
counts are **not** a matched-latency architecture ablation.

Image PSNR/SSIM are computed on full mod-cropped validation images, raw floats,
with the configured common shave. They are averaged per image. LPIPS is optional
and follows the main package's clipping/normalization protocol; install with
`python -m pip install -e ".[research,metrics]"` before enabling it.

Component MSE/R² are sample-weighted using a fixed uniform sample per image,
shared across contexts and predictors. RGB channels are numbered 0/1/2;
components are numbered 1..15. Zero-variance R² is left empty. Both the baseline
using the **training mean** and the zero-coefficient baseline are reported.
`gain_vs_zero` is useful when deciding whether retaining a PCA direction helps.
Do not confuse sampled component statistics with full-image error aggregation.

All probe prefixes use one full-rank predictor. Smaller prefixes therefore do
not constitute separately optimized/scaled deployment models. `benchmark.json`
measures their **full-rank** runtime. Use `sweep` to measure independent models
that really change deployment rank/capacity.

Latency is batch-one fp32, synchronized wall time on CUDA, with warmup and repeated
measurements; GPU transfer and preprocessing are excluded. Predictor, decoder
plus repeated LR base, and full model are timed separately. The separate medians
need not sum to the end-to-end median. MAC estimates include convolution products
and PCA synthesis, excluding additions, activations and per-coefficient scaling.
CUDA process peak allocation and additional allocation above the pre-inference
baseline are reported (other live tensors can affect the process peak);
CPU peak memory is not measured. Set identical
benchmark input sizes/hardware when comparing experiments.

## Outputs

```text
output/
  protocol.json, environment.json
  basis/full.pt, fingerprint.json, train_coefficients.pt
  oracle/<method>/
  ridge/<method>/<channel_mode>_r<context>/
  cnn/<method>/rgb_r<context>_seed<seed>/
  sweep/<method>/<variant>_k<rank>_seed<seed>/
  reports/
    r2_vs_context.csv
    coefficient_energy.csv
    oracle_vs_rank.csv
    quality_vs_rank.csv
    cost.csv
    backbone_sweep.csv       # after optional sweep
    backbone_summary.csv     # per variant/rank: seed mean, population std, Pareto flag
    pareto_<method>.png
    r2_<method>_<probe>_<mode>.png
    oracle_vs_rank.png
```

Per-run evaluation includes `per_image.csv`, `quality.csv`, `components.csv`
and `summary.json`. Probe run checkpoints contain their own model type metadata
(context, width/channel mode and full rank); they are research artifacts, not
direct inputs to the main `test.py`. Sweep checkpoints are standard train/test
checkpoints. Keep test datasets separate from the validation set used to choose
contexts, modules and hyperparameters.

This script does not yet implement region-conditioned errors, effective receptive
field saliency, learned basis rotation, or a predictability-driven allocation
rule. Its diagnostics and optional capacity sweep provide evidence for those
later steps without treating untested modules as improvements.
