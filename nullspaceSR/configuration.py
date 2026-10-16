"""Translate experiment YAML sections into command arguments."""
from copy import deepcopy


def command_defaults(config, stage, allowed):
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a mapping")
    sections = {"model", "data", "train", "pca", "evaluation", "runtime", "logging", "metadata"}
    if not (set(config) & sections):
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"Unknown YAML keys: {sorted(unknown)}")
        return config
    unknown = set(config) - sections
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    for name, value in config.items():
        if not isinstance(value, dict):
            raise ValueError(f"Section {name} must be a mapping")
    defaults = {}
    keys = {
        "model": {"scale", "rank", "width", "basis", "checkpoint", "null_space", "channels"},
        "data": {"train_hr", "val_hr", "hr_dir", "train_dataset", "val_dataset",
                 "test_dataset", "train_dataset_options", "val_dataset_options",
                 "test_dataset_options", "crop_size", "repeat", "augment", "workers", "resize"},
        "train": {"output", "resume", "epochs", "batch_size", "lr", "min_lr",
                  "weight_decay", "grad_clip", "val_every", "log_every", "checkpoint_every"},
        "pca": {"output", "batch_size", "repeat", "augment"},
        "evaluation": {"output", "shave", "save_images"},
        "runtime": {"seed", "device"},
        "logging": {"enabled", "project", "entity", "name", "mode", "tags"},
    }
    for section, accepted in keys.items():
        values = config.get(section, {})
        unknown = set(values) - accepted
        if unknown:
            raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")
    active = ["model", "data", stage, "runtime"]
    if stage == "train":
        defaults.update({k: v for k, v in config.get("evaluation", {}).items()
                         if k == "shave"})
    for section in active:
        defaults.update({k: deepcopy(v) for k, v in config.get(section, {}).items()
                         if k in allowed})
    logging_names = {"enabled": "wandb_enabled", "project": "wandb_project",
                     "entity": "wandb_entity", "name": "wandb_name",
                     "mode": "wandb_mode", "tags": "wandb_tags"}
    defaults.update({logging_names[key]: deepcopy(value)
                     for key, value in config.get("logging", {}).items()
                     if logging_names[key] in allowed})
    resize = config.get("data", {}).get("resize")
    if resize is not None:
        if not isinstance(resize, dict):
            raise ValueError("data.resize must be a mapping")
        for split in {"train": ("train", "val"), "pca": ("train",),
                      "evaluation": ("test",)}.get(stage, ()):
            if defaults.get(f"{split}_dataset", "div2k") not in {"div2k", "hr_folder"}:
                continue
            key = f"{split}_dataset_options"
            options = defaults.get(key) or {}
            if not isinstance(options, dict):
                raise ValueError(f"data.{key} must be a mapping")
            if not isinstance(options.get("resize", {}), dict):
                raise ValueError(f"data.{key}.resize must be a mapping")
            options["resize"] = {**deepcopy(resize), **options.get("resize", {})}
            defaults[key] = options
    return defaults
