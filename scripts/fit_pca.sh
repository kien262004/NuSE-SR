#!/usr/bin/env bash
cd "$(dirname "$0")/.." && python3 fit_pca.py --config configs/pca_div2k.yaml
