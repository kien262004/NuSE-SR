"""Custom dataset fixture with no files/root/training/set_epoch attributes."""
import torch
from torch.utils.data import Dataset
from nullspaceSR.data import make_sample


class ToySRDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split

    def __len__(self):
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + (0 if self.split == "train" else 20))
        hr = torch.rand(1, 8, 12, generator=generator)
        lr = torch.nn.functional.avg_pool2d(hr, 2)
        return make_sample(lr, hr, sample_id=f"{self.split}_{index}",
                           meta={"origin": "in_memory", "split": self.split})
