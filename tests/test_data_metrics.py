import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from nullspaceSR.data import DIV2KDataset, make_loader
from nullspaceSR.metrics import rgb_mse, rgb_psnr, rgb_ssim


class DataMetricTests(unittest.TestCase):
    def test_dataset_consistency_reproducibility_and_modcrop(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            pixels = bytes(i % 256 for i in range(19 * 23 * 3))
            Image.frombytes("RGB", (23, 19), pixels).save(folder / "0001.png")
            train = DIV2KDataset(folder, scale=4, training=True, crop_size=16,
                                 repeat=2, augment=True, seed=7)
            a, b = train[0], train[0]
            torch.testing.assert_close(a["target"], b["target"])
            torch.testing.assert_close(torch.nn.functional.avg_pool2d(a["target"], 4), a["input"])
            self.assertEqual(tuple(a["target"].shape), (3, 16, 16))
            train.set_epoch(1)
            self.assertFalse(torch.equal(a["target"], train[0]["target"]))
            val = DIV2KDataset(folder, scale=4)
            self.assertEqual(tuple(val[0]["target"].shape), (3, 16, 20))
            batch = next(iter(make_loader(val, batch_size=1)))
            self.assertEqual(tuple(batch["input"].shape), (1, 3, 4, 5))

    def test_metric_values_and_border(self):
        target = torch.zeros(2, 3, 16, 20)
        predicted = torch.full_like(target, 0.1)
        torch.testing.assert_close(rgb_mse(predicted, target), torch.full((2,), 0.01, dtype=torch.float64))
        torch.testing.assert_close(rgb_psnr(predicted, target), torch.full((2,), 20.0, dtype=torch.float64))
        self.assertTrue(torch.isinf(rgb_psnr(target, target)).all())
        torch.testing.assert_close(rgb_ssim(target, target), torch.ones(2))
        border = target.clone()
        border[..., 0, :] = 1
        torch.testing.assert_close(rgb_mse(border, target, shave=1), torch.zeros(2, dtype=torch.float64))
        with self.assertRaises(ValueError):
            rgb_mse(target, target, shave=8)


if __name__ == "__main__":
    unittest.main()
