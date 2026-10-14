#!/usr/bin/env bash
cd "$(dirname "$0")/.." && python3 train.py --config configs/train_div2k.yaml
