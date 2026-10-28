"""Seed, ghi JSON / CSV."""
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def safe(v):
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [safe(x) for x in v]
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, torch.Tensor):
            return safe(v.tolist())
        return v
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    for row in rows[1:]:
        keys += [k for k in row if k not in keys]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
