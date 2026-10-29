"""Configuration, reproducibility, checkpoint and output helpers."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from .configuration import command_defaults


def parse_args(parser: argparse.ArgumentParser, *, stage=None):
    """Read experiment sections or legacy flat YAML; CLI flags override defaults.

    Repository entry points set the working directory to the project root before parsing paths.
    """
    parser.add_argument("--config", help="Experiment YAML configuration")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with open(preliminary.config, encoding="utf-8") as stream:
            defaults = yaml.safe_load(stream) or {}
        allowed = {action.dest for action in parser._actions} - {"help", "config"}
        try:
            defaults = command_defaults(defaults, stage, allowed)
        except ValueError as error:
            parser.error(str(error))
        parser.set_defaults(**defaults)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_rgb(path, image):
    """Clamp/round for PNG preview ONLY; metrics use raw model outputs."""
    image = image.detach().cpu().clamp(0, 1).mul(255).round().byte()
    h, w = image.shape[-2:]
    pixels = image.permute(1, 2, 0).contiguous().numpy().tobytes()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    modes = {1: "L", 3: "RGB", 4: "RGBA"}
    if image.shape[0] not in modes:
        raise ValueError("PNG export supports 1, 3 or 4 image channels")
    Image.frombytes(modes[image.shape[0]], (w, h), pixels).save(path)


def save_checkpoint(path, state):
    """Atomic replacement prevents partial checkpoints after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def model_from_checkpoint(checkpoint):
    from .models import NuSESR, build_predictor
    from .pca import BlockNullSpacePCA

    arch = checkpoint["architecture"]
    state = checkpoint["model"]
    def predictor(decoder=None):
        return build_predictor(arch.get("backbone", "imdn"), arch["channels"],
                               arch["rank"] if decoder is not None else None, arch["width"],
                               scale=arch["scale"], num_blocks=arch.get("num_blocks", 7),
                               block=arch.get("block", "plain"), budgets=arch.get("budgets") or None,
                               stage_blocks=arch.get("stage_blocks"), decoder=decoder)
    # Older checkpoints always used PCA and have no null_space flag.
    if not arch.get("null_space", True):
        model = NuSESR(predictor())
        model.load_state_dict(state)
        return model
    eigenvalues = state["decoder.eigenvalues"]
    decoder = BlockNullSpacePCA(arch["scale"], state["decoder.mean"],
                               state["decoder.components"],
                               eigenvalues if eigenvalues.numel() else None)
    if decoder.channels != arch["channels"] or decoder.rank != arch["rank"]:
        raise ValueError("Checkpoint architecture and PCA buffers do not match")
    model = NuSESR(predictor(decoder), decoder)
    model.load_state_dict(state)
    return model
