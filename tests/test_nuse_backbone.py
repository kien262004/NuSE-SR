import copy
import unittest
from unittest.mock import patch

import torch

from nullspaceSR.models import IMDN, NuSEBackbone, NuSESR, build_predictor
from nullspaceSR.models.nuse import coefficient_layout
from nullspaceSR.pca import BlockNullSpacePCA
from nullspaceSR.utils import model_from_checkpoint


class NuSEBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_group_order_and_prefix_execution(self):
        predictor = NuSEBackbone(channels=2, rank=5, width=4, num_blocks=3,
                                 budgets=[2, 5], stage_blocks=[2, 1])
        with torch.no_grad():
            predictor.heads[0].readout.bias.copy_(torch.tensor([1., 2., 11., 12.]))
            predictor.heads[1].readout.bias.copy_(torch.tensor([3., 4., 5., 13., 14., 15.]))
        lr = torch.rand(1, 2, 3, 5)
        full = predictor(lr).reshape(1, 2, 5, 3, 5)
        torch.testing.assert_close(full[0, :, :, 0, 0],
                                   torch.tensor([[1., 2., 3., 4., 5.], [11., 12., 13., 14., 15.]]))
        with patch.object(predictor.stages[1], "forward", side_effect=AssertionError("late stage ran")):
            early = predictor(lr, k=2).reshape(1, 2, 2, 3, 5)
        torch.testing.assert_close(early, full[:, :, :2])

    def test_pca_consistency_gradient_and_checkpoint(self):
        hr = torch.rand(1, 3, 16, 20)
        decoder = BlockNullSpacePCA.fit([hr], scale=2, rank=2)
        for kind in ("plain", "rep", "lk7"):
            predictor = build_predictor("nuse", 3, 2, 4, scale=2, num_blocks=3,
                                        block=kind, budgets=[1, 2], stage_blocks=[2, 1], decoder=decoder)
            model = NuSESR(predictor, decoder)
            lr = torch.nn.functional.interpolate(hr, scale_factor=.5, mode="bicubic",
                                                  align_corners=False, antialias=True)
            optimizer = torch.optim.Adam(predictor.parameters(), lr=.001)
            for _ in range(2):
                optimizer.zero_grad()
                loss = (model(lr) - hr).square().mean()
                loss.backward()
                optimizer.step()
            self.assertGreater(predictor.stem.weight.grad.abs().sum().item(), 0)
            torch.testing.assert_close(model.operator(model(lr)), lr, atol=1e-6, rtol=1e-6)
            self.assertEqual(list(decoder.parameters()), [])
            architecture = dict(backbone="nuse", channels=3, rank=2, width=4, scale=2,
                                null_space=True, num_blocks=3, block=kind,
                                budgets=[1, 2], stage_blocks=[2, 1])
            restored = model_from_checkpoint(dict(architecture=architecture, model=model.state_dict()))
            torch.testing.assert_close(restored(lr), model(lr), atol=0, rtol=0)

    def test_image_mode_shapes_and_checkpoint(self):
        for scale in (2, 3, 4):
            model = NuSESR(NuSEBackbone(channels=1, width=4, num_blocks=2, scale=scale))
            lr = torch.rand(2, 1, 3, 5)
            self.assertEqual(tuple(model(lr).shape), (2, 1, 3 * scale, 5 * scale))
            model(lr).square().mean().backward()
            self.assertGreater(model.predictor.stem.weight.grad.abs().sum().item(), 0)
            architecture = dict(backbone="nuse", channels=1, rank=None, width=4, scale=scale,
                                null_space=False, num_blocks=2, block="plain", budgets=[], stage_blocks=None)
            restored = model_from_checkpoint(dict(architecture=architecture, model=model.state_dict()))
            torch.testing.assert_close(restored(lr), model(lr), atol=0, rtol=0)

    def test_rep_fusion_preserves_every_prefix(self):
        predictor = NuSEBackbone(channels=3, rank=5, width=4, num_blocks=3, block="rep",
                                 budgets=[2, 5], stage_blocks=[2, 1]).eval()
        with torch.no_grad():
            for parameter in predictor.parameters():
                parameter.normal_(0, .1)
        fused = copy.deepcopy(predictor).fuse().fuse()
        lr = torch.rand(1, 3, 5, 7)
        for k in (2, 5):
            torch.testing.assert_close(fused(lr, k), predictor(lr, k), atol=1e-6, rtol=1e-5)
        self.assertLess(sum(p.numel() for p in fused.parameters()),
                        sum(p.numel() for p in predictor.parameters()))

    def test_spectral_layout_and_validation(self):
        energy = torch.tensor([[10., 10., 4., 2., 1.]]).repeat(3, 1)
        budgets, blocks = coefficient_layout(5, energy, 7)
        self.assertNotIn(1, budgets)  # Never split the repeated top eigenvalue.
        self.assertEqual(budgets[-1], 5)
        self.assertEqual(sum(blocks), 7)
        self.assertTrue(all(n > 0 for n in blocks))
        self.assertEqual(coefficient_layout(5, torch.zeros_like(energy), 7), ([5], [7]))
        self.assertEqual(coefficient_layout(5, None, 7), ([5], [7]))
        self.assertEqual(coefficient_layout(5, energy * 1e-20, 7), (budgets, blocks))
        self.assertEqual(coefficient_layout(5, torch.tensor([[1., 0., 0., 0., 0.]]), 7)[0][-1], 5)
        for kwargs in ({"budgets": [2, 4]}, {"budgets": [2, 5], "stage_blocks": [0, 7]},
                       {"budgets": [2, 5], "stage_blocks": [1, 1]}, {"block": "invalid"},
                       {"eigenvalues": torch.ones(2, 5)}):
            with self.assertRaises(ValueError):
                NuSEBackbone(rank=5, **kwargs)

    def test_legacy_image_checkpoint(self):
        model = NuSESR(IMDN(channels=1, width=4, scale=2))
        architecture = dict(name="IMDN", channels=1, rank=None, width=4, scale=2, null_space=False)
        restored = model_from_checkpoint(dict(architecture=architecture, model=model.state_dict()))
        self.assertIsInstance(restored.predictor, IMDN)
        lr = torch.rand(1, 1, 3, 5)
        torch.testing.assert_close(restored(lr), model(lr), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
