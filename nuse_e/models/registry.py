"""Tạo model theo tên và nạp checkpoint (của pipeline này hoặc của thí nghiệm cũ)."""
import torch

from .baselines import ImageSR, LegacyNuSEPlugin
from .nuse_e import NuSEE

def build_model(name, method, stats=None, arch=None):
    if name.startswith("NuSE-E"):                        # "NuSE-E" or "NuSE-E/<variant>"
        return NuSEE(stats, arch)
    return ImageSR(name)


def budgets_of(model):
    return list(model.budgets) if isinstance(model, NuSEE) else [None]


def load_run(path, stats=None, arch=None):
    """Load a checkpoint written by train_run.py or a previous-experiment best.pt."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "nuse_e" in ck:                                   # this notebook
        info = ck["nuse_e"]
        if info["model"].startswith("NuSE-E"):
            model = NuSEE(ck["stats"], ck["arch"])
            model.load_state_dict(ck["model"])
            model.fuse()
        else:
            model = ImageSR(info["model"])
            model.load_state_dict(ck["model"])
        return model.eval(), ck
    arch_old = ck["architecture"]                        # previous experiments
    names = {"spanv2": "SPANV2", "disp": "DISP", "plksr_tiny": "PLKSR-Tiny"}
    name = names[arch_old["backbone"]]
    if arch_old.get("null_space", False):
        model = LegacyNuSEPlugin(name, ck["model"])
        model.load_state_dict(ck["model"])
    else:
        model = ImageSR(name)
        model.load_state_dict(ck["model"])
    return model.eval(), ck
