# nullspaceSR package guide

The Python package lives in `nullspaceSR/`. Project commands live at the repository root:

```text
train.py       # training and resume
test.py        # checkpoint evaluation
fit_pca.py     # offline PCA fitting entry point
scripts/       # fit_pca.sh, train.sh, test.sh launchers
nullspaceSR/
  data/        # datasets, sample contract, registry, loaders
  models/     # coefficient predictor and NuSESR
  operators/  # measurement operator and pseudoinverse
  engine.py   # shared training loop
  evaluation.py
  metrics.py
  patches.py
  pca.py
  losses.py
  utils.py
```

## Install and run

Run from the repository root:

```bash
python -m pip install -e .
python examples/smoke_sr.py
python -m unittest discover -s tests
```

The DIV2K workflow also uses root-level scripts and configs:

```bash
python fit_pca.py --config configs/pca_div2k.yaml
python train.py --config configs/train_div2k.yaml
python test.py --config configs/test_div2k.yaml
```

Installed command aliases are `nusesr-fit-pca`, `nusesr-train`, and `nusesr-test`.

## Build a model

```python
from nullspaceSR import BlockNullSpacePCA, NuSESR, IMDN

pca = BlockNullSpacePCA.fit(train_hr_loader, scale=4, rank=8)
model = NuSESR(IMDN(channels=3, rank=8, width=64), pca)
sr = model(lr)
null_detail = model.predict_null(lr)
```

Replace the coefficient predictor while keeping its output shape `[B, C*k, H_LR, W_LR]`. Fit PCA offline, then load its basis with `BlockNullSpacePCA.load(...)`.

The shared dataset interface is documented in [data_contract.md](data_contract.md). DIV2K data files and generated outputs are configured through the YAML files under `configs/`.
