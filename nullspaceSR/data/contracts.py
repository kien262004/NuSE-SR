"""Dataset-independent SR samples and batches."""
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SRSample(TypedDict):
    input: Tensor                 # CHW measurement / LR
    target: Tensor | None         # CHW HR; None for inference without labels
    id: str                       # Stable unique sample ID
    meta: dict[str, Any]           # Paths, preprocessing, degradation, etc.
    extras: dict[str, Any]         # Optional dataset-specific payload


def make_sample(input: Tensor, target: Tensor | None = None, *,
                sample_id: str, meta=None, extras=None) -> SRSample:
    """Create and validate a sample. Caller controls preprocessing/range."""
    result = {"input": input, "target": target, "id": sample_id,
              "meta": dict(meta or {}), "extras": dict(extras or {})}
    validate_sample(result)
    return result


def validate_sample(sample):
    if not isinstance(sample, Mapping):
        raise TypeError("Dataset must return a mapping; use MappingDatasetAdapter for legacy keys")
    missing = {"input", "id"} - sample.keys()
    if missing:
        raise ValueError(f"Missing sample fields: {sorted(missing)}")
    for key in ("input", "target"):
        tensor = sample.get(key)
        if key == "target" and tensor is None:
            continue
        if not isinstance(tensor, Tensor) or tensor.ndim != 3 or not tensor.is_floating_point():
            raise ValueError(f"{key} must be a floating CHW tensor")
        if min(tensor.shape) < 1:
            raise ValueError(f"{key} cannot have empty dimensions")
        if tensor.device.type != "cpu":
            raise ValueError("Dataset tensors must be on CPU; the engine handles device transfer")
    if not isinstance(sample["id"], str) or not sample["id"]:
        raise ValueError("Sample id must be a nonempty string")
    for key in ("meta", "extras"):
        if not isinstance(sample.get(key, {}), Mapping):
            raise ValueError(f"{key} must be a mapping")


def collate_sr(samples: Sequence[Mapping]):
    """Stack tensors; keep IDs/metadata/extras as ordered per-sample lists.

    No implicit padding/resizing: these would change the forward model.
    Use batch_size=1 for variable-size images, or crop/bucket in the dataset.
    Unlabelled and labelled samples cannot be mixed within one batch.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    for sample in samples:
        validate_sample(sample)
    targets = [sample.get("target") for sample in samples]
    if any(target is None for target in targets) and not all(target is None for target in targets):
        raise ValueError("Cannot mix labelled and unlabelled samples in one batch")
    def stack(key):
        tensors = [sample[key] for sample in samples]
        if len({(tuple(t.shape), t.dtype) for t in tensors}) != 1:
            raise ValueError(f"{key} shapes/dtypes differ; use batch_size=1 or aligned crops")
        return torch.stack(tensors)
    return {"input": stack("input"),
            "target": None if targets[0] is None else stack("target"),
            "id": [sample["id"] for sample in samples],
            "meta": [dict(sample.get("meta", {})) for sample in samples],
            "extras": [dict(sample.get("extras", {})) for sample in samples]}


def supervised_batch(batch, device, *, channels=None, scale=None, operator=None):
    """Validate model compatibility, transfer tensors, return (input,target).

    Channels/scale are model compatibility checks. Measurement consistency is
    opt-in via operator; LR degradation can be independent of the model operator.
    """
    lr, hr = batch["input"], batch.get("target")
    if hr is None:
        raise ValueError("Supervised training/evaluation/PCA fitting requires target HR")
    if lr.ndim != 4 or hr.ndim != 4 or not lr.is_floating_point() or not hr.is_floating_point():
        raise ValueError("Batch input/target must be floating BCHW tensors")
    if lr.shape[:2] != hr.shape[:2]:
        raise ValueError("Input/target batch size and channels must match")
    if channels is not None and lr.shape[1] != channels:
        raise ValueError(f"Dataset channels do not match decoder channels={channels}")
    if scale is not None and hr.shape[-2:] != (lr.shape[-2] * scale, lr.shape[-1] * scale):
        raise ValueError("Input/target spatial dimensions do not match model scale")
    lr, hr = lr.to(device=device, dtype=torch.float32), hr.to(device=device, dtype=torch.float32)
    if not torch.isfinite(lr).all() or not torch.isfinite(hr).all():
        raise ValueError("Input/target contains non-finite values")
    if operator is not None and not torch.allclose(operator(hr), lr, atol=1e-5, rtol=1e-5):
        raise ValueError(
            f"Dataset input is not A(target) for {type(operator).__name__}; "
            "explicit measurement validation requires matching degradation/normalization. "
            "Omit operator when LR generation is independent of the model operator."
        )
    return lr, hr


class MappingDatasetAdapter(Dataset):
    """Map existing dictionary datasets to the shared contract without edits."""
    def __init__(self, dataset, *, input_key="lr", target_key="hr", id_key="name"):
        self.dataset = dataset
        self.input_key, self.target_key, self.id_key = input_key, target_key, id_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return make_sample(item[self.input_key], item.get(self.target_key),
                           sample_id=str(item.get(self.id_key, index)),
                           meta=item.get("meta"),
                           extras={key: value for key, value in item.items()
                                   if key not in {self.input_key, self.target_key, self.id_key, "meta"}})

    def set_epoch(self, epoch):
        set_dataset_epoch(self.dataset, epoch)

    def manifest(self):
        return dataset_manifest(self.dataset)


def set_dataset_epoch(dataset, epoch):
    """Optional hook; ordinary torch datasets need not implement it."""
    callback = getattr(dataset, "set_epoch", None)
    if callable(callback):
        callback(epoch)


def dataset_manifest(dataset):
    """Optional provenance hook; no reliance on dataset.files/root/training."""
    callback = getattr(dataset, "manifest", None)
    if callable(callback):
        result = callback()
        if not isinstance(result, Mapping):
            raise ValueError("dataset.manifest() must return a mapping")
        return dict(result)
    return {"dataset": f"{type(dataset).__module__}:{type(dataset).__qualname__}",
            "length": len(dataset), "sources": []}
