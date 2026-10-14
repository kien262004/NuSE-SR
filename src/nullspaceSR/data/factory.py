"""Dataset registry plus module:Class imports for custom data sources."""
import importlib
import json
from collections.abc import Mapping

from torch.utils.data import Dataset
from .div2k import DIV2KDataset

_DATASETS = {"div2k": DIV2KDataset, "hr_folder": DIV2KDataset}


def register_dataset(name, constructor):
    if name in _DATASETS:
        raise ValueError(f"Dataset already registered: {name}")
    _DATASETS[name] = constructor


def build_dataset(name, options=None, *, folder_defaults=None):
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("Dataset options must be a mapping")
    if name in _DATASETS:
        constructor = _DATASETS[name]
    elif ":" in name:
        module, symbol = name.split(":", 1)
        constructor = getattr(importlib.import_module(module), symbol)
    else:
        raise ValueError(f"Unknown dataset {name!r}; register it or use module:Class")
    kwargs = dict(folder_defaults or {}) if name in {"div2k", "hr_folder"} else {}
    kwargs.update(options or {})
    if name in {"div2k", "hr_folder"} and not kwargs.get("hr_dir"):
        raise ValueError("HR folder dataset requires hr_dir (or the CLI HR directory argument)")
    dataset = constructor(**kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset constructor must return a torch.utils.data.Dataset")
    if not len(dataset):
        raise ValueError("Dataset is empty")
    return dataset


def add_dataset_arguments(parser, split):
    parser.add_argument(f"--{split}-dataset", default="div2k", help="Registry name or module:Class")
    parser.add_argument(f"--{split}-dataset-options", type=json.loads, default=None,
                        help="Constructor kwargs as JSON (or a mapping in YAML)")
