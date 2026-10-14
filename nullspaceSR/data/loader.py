import torch
from torch.utils.data import DataLoader

from ..utils import seed_worker
from .contracts import collate_sr


def make_loader(dataset, *, batch_size=8, workers=0, shuffle=False, seed=2026,
                pin_memory=False):
    """Any torch Dataset returning SRSample; no DIV2K-specific attributes."""
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive; workers must be nonnegative")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin_memory,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=collate_sr, persistent_workers=False, drop_last=False)
