"""Mathematical invariants, including rank-deficient PCA and no projection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nullspaceSR import (
    AveragePoolSR, BlockNullSpacePCA, NuSESR,
    IMDN, block_null_basis, coefficient_image_mse,
)
from nullspaceSR.utils import model_from_checkpoint
from nullspaceSR.configuration import command_defaults


class NullSpaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_image_head_and_explicit_config_switch(self):
        defaults = command_defaults({"model": {"null_space": False, "channels": 1}},
                                    "train", {"null_space", "channels"})
        self.assertEqual(defaults, {"null_space": False, "channels": 1})
        for scale in (2, 3, 4):
            model = NuSESR(IMDN(channels=1, width=4, scale=scale))
            lr = torch.rand(2, 1, 3, 5)
            sr = model(lr)
            self.assertEqual(tuple(sr.shape), (2, 1, 3 * scale, 5 * scale))
            self.assertIsNone(model.operator)
            sr.square().mean().backward()
            self.assertIsNotNone(model.predictor.image_head[0].weight.grad)
            with self.assertRaisesRegex(RuntimeError, "plugin"):
                model.predict_coefficients(lr)

    def test_legacy_checkpoint_without_mode_flag(self):
        decoder = BlockNullSpacePCA.fit([torch.rand(1, 3, 8, 8)], scale=2, rank=2)
        model = NuSESR(IMDN(3, 2, width=4), decoder)
        restored = model_from_checkpoint({
            "architecture": {"scale": 2, "rank": 2, "channels": 3, "width": 4},
            "model": model.state_dict()})
        lr = torch.rand(1, 3, 4, 4)
        torch.testing.assert_close(restored(lr), model(lr))

    def test_analytic_basis(self):
        for scale in (2, 4):
            Q = block_null_basis(scale)
            torch.testing.assert_close(Q.T @ Q, torch.eye(scale**2 - 1, dtype=Q.dtype))
            torch.testing.assert_close(Q.sum(0), torch.zeros(scale**2 - 1, dtype=Q.dtype), atol=1e-14, rtol=0)

    def test_low_rank_decomposition_and_gradient(self):
        for scale, rank in ((2, 2), (4, 8)):
            with self.subTest(scale=scale):
                decoder = BlockNullSpacePCA.fit([torch.rand(2, 3, 16, 24)], scale, rank)
                model = NuSESR(IMDN(3, rank, width=8), decoder)
                hr = torch.rand(2, 3, 16, 24)
                lr = model.operator(hr)
                target = decoder.encode_hr(hr)
                predicted = model.predict_coefficients(lr)
                output = model.operator.pseudoinverse(lr) + decoder(predicted)
                oracle = model.operator.pseudoinverse(lr) + decoder(target)
                loss = coefficient_image_mse(predicted, target, channels=3, scale=scale)
                torch.testing.assert_close((output - hr).square().mean(),
                                           (oracle - hr).square().mean() + loss)
                torch.testing.assert_close(model.operator(output), lr, atol=1e-6, rtol=1e-6)
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.predictor.parameters()))
                self.assertEqual(list(decoder.parameters()), [])
                # Forward must not call A to project the predictor output.
                with patch.object(model.operator, "forward", side_effect=AssertionError("Unexpected projection")):
                    torch.testing.assert_close(model(lr), output)

    def test_full_rank_with_rank_deficient_training_data(self):
        # All training blocks are constant: covariance carries no detail.
        decoder = BlockNullSpacePCA.fit([torch.ones(1, 1, 8, 8)], scale=4, rank=15)
        op = AveragePoolSR(4)
        hr = torch.rand(2, 1, 12, 20)
        reconstructed = op.pseudoinverse(op(hr)) + decoder(decoder.encode_hr(hr))
        torch.testing.assert_close(reconstructed, hr, atol=1e-6, rtol=1e-6)
        coeffs = torch.randn(2, 15, 3, 5)
        torch.testing.assert_close(op(decoder(coeffs)), torch.zeros(2, 1, 3, 5), atol=1e-6, rtol=0)

    def test_basis_save_load(self):
        decoder = BlockNullSpacePCA.fit([torch.rand(1, 3, 8, 8)], scale=2, rank=2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "basis.pt"
            decoder.save(path)
            loaded = BlockNullSpacePCA.load(path)
        coeffs = torch.randn(2, 6, 4, 7)
        torch.testing.assert_close(decoder(coeffs), loaded(coeffs))

    def test_predictor_shape_guard(self):
        decoder = BlockNullSpacePCA.fit([torch.rand(1, 3, 8, 8)], scale=2, rank=2)
        model = NuSESR(torch.nn.Identity(), decoder)
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
