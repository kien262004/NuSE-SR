import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch
from PIL import Image

from nullspaceSR.data import DIV2KDataset, make_loader
from nullspaceSR.metrics import rgb_mse, rgb_psnr, rgb_ssim, LPIPSMetric
from nullspaceSR.models import IMDN, NuSESR
from nullspaceSR.pca import BlockNullSpacePCA
from nullspaceSR.evaluation import evaluate
from nullspaceSR.data import make_sample
from nullspaceSR.configuration import command_defaults


class DataMetricTests(unittest.TestCase):
    def test_lpips_preprocessing_freezing_and_rng(self):
        class RecordingNetwork(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(1))
                self.settings = kwargs
            def forward(self, x, y):
                self.inputs = (x, y)
                return (x - y).square().mean((1, 2, 3), keepdim=True)

        with patch.dict('sys.modules', {'lpips': SimpleNamespace(LPIPS=RecordingNetwork)}):
            rng = torch.get_rng_state()
            metric = LPIPSMetric()
            torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertFalse(metric.network.training)
        self.assertFalse(metric.network.weight.requires_grad)
        self.assertEqual(metric.network.settings['net'], 'alex')
        predicted = torch.full((2, 1, 34, 38), 2.0, requires_grad=True)
        target = torch.full_like(predicted, -1.0)
        values = metric(predicted, target, shave=1)
        torch.testing.assert_close(values, torch.full((2,), 4.0))
        x, y = metric.network.inputs
        self.assertEqual(tuple(x.shape), (2, 3, 32, 36))
        self.assertTrue((x == 1).all())
        self.assertTrue((y == -1).all())
        self.assertFalse(values.requires_grad)
        with self.assertRaisesRegex(ValueError, '32x32'):
            metric(predicted, target, shave=2)
        with self.assertRaisesRegex(ValueError, 'RGB or grayscale'):
            metric(torch.zeros(1, 4, 32, 32), torch.zeros(1, 4, 32, 32))

    def test_lpips_evaluation_and_config_for_both_modes(self):
        class ConstantMetric:
            def __call__(self, predicted, target, shave):
                self.shave = shave
                return torch.tensor([0.25])
        defaults = command_defaults({'evaluation': {'lpips': False, 'shave': 1}},
                                    'train', {'lpips', 'shave'})
        self.assertEqual(defaults, {'lpips': False, 'shave': 1})
        hr = torch.rand(3, 32, 36)
        decoder = BlockNullSpacePCA.fit([hr.unsqueeze(0)], scale=2, rank=2)
        lr = torch.nn.functional.avg_pool2d(hr, 2)
        sample = make_sample(lr, hr, sample_id='test')
        loader = make_loader([sample], batch_size=1)
        models = [NuSESR(IMDN(width=4, scale=2)),
                  NuSESR(IMDN(rank=2, width=4), decoder)]
        for model in models:
            metric = ConstantMetric()
            summary, rows = evaluate(model, loader, 'cpu', shave=1, lpips_metric=metric)
            self.assertEqual(summary['lpips'], 0.25)
            self.assertEqual(rows[0]['lpips'], 0.25)
            self.assertIn('rgb_psnr', summary)
            self.assertIn('rgb_ssim', summary)
            self.assertEqual(metric.shave, 1)
            self.assertTrue(model.training)
            self.assertIn('LPIPS v0.1 AlexNet', summary['lpips_protocol'])

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
