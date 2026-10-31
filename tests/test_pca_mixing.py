"""Regression coverage for the PCA-15 to rank-8 port."""
import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from nullspaceSR.models import IMDN, NuSESR
from nullspaceSR.pca import BlockNullSpacePCA, PCAMixingBasis
from nullspaceSR.utils import model_from_checkpoint


class PCAMixingTest(unittest.TestCase):
    def test_initialization_gradients_and_data_consistency(self):
        torch.manual_seed(2026)
        source = BlockNullSpacePCA(4, torch.zeros(3, 15),
                                   torch.eye(15).expand(3, -1, -1).clone())
        mixing = PCAMixingBasis.from_pca(source, rank=8, source_rank=15)
        fixed = BlockNullSpacePCA(4, source.mean, source.components[..., :8])
        torch.testing.assert_close(mixing.effective_basis(), fixed.basis, atol=2e-6, rtol=2e-6)
        self.assertEqual(tuple(mixing.raw_mixing.shape), (3, 8, 15))
        model = NuSESR(IMDN(3, 8, 4), mixing)
        lr = torch.rand(1, 3, 4, 4)
        hr = torch.rand(1, 3, 16, 16)
        out = model(lr)
        self.assertEqual(tuple(out.shape), tuple(hr.shape))
        self.assertLess((model.operator(out) - lr).abs().max().item(), 1e-5)
        loss = (out - hr).square().mean()
        loss.backward()
        self.assertGreater(mixing.raw_mixing.grad.abs().sum().item(), 0)
        old = mixing.raw_mixing.detach().clone()
        optimizer = torch.optim.AdamW([
            {"params": model.predictor.parameters(), "lr": 1e-3},
            {"params": mixing.parameters(), "lr": 1e-4}], weight_decay=0)
        optimizer.step()
        self.assertGreater((mixing.raw_mixing - old).abs().max().item(), 0)
        gram = mixing.effective_basis().transpose(-1, -2) @ mixing.effective_basis()
        torch.testing.assert_close(gram, torch.eye(8).expand(3, -1, -1), atol=1e-5, rtol=1e-5)
        checkpoint = {"architecture": {"null_space": True, "basis_mode": "pca_mixing",
                                       "scale": 4, "rank": 8, "source_rank": 15,
                                       "channels": 3, "width": 4}, "model": model.state_dict()}
        restored = model_from_checkpoint(checkpoint)
        torch.testing.assert_close(restored(lr), model(lr), atol=1e-6, rtol=1e-6)

    def test_two_epoch_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                folder = root / split
                folder.mkdir()
                for index in range(2 if split == "train" else 1):
                    pixels = bytes((i * 13 + index * 17) % 256 for i in range(16 * 20 * 3))
                    Image.frombytes("RGB", (20, 16), pixels).save(folder / f"{index:04d}.png")
            source = BlockNullSpacePCA(2, torch.zeros(3, 3),
                                       torch.eye(3).expand(3, -1, -1).clone())
            basis = root / "source.pt"
            source.save(basis)
            def run(*extra):
                result = subprocess.run([
                    sys.executable, "-m", "train", "--train-hr", str(root / "train"),
                    "--val-hr", str(root / "val"), "--basis", str(basis),
                    "--scale", "2", "--rank", "2", "--source-rank", "3",
                    "--basis-mode", "pca_mixing", "--width", "4", "--crop-size", "16",
                    "--repeat", "1", "--batch-size", "2", "--epochs", "2",
                    "--checkpoint-every", "1", "--device", "cpu", "--no-lpips",
                    *map(str, extra)], cwd=Path(__file__).resolve().parents[1],
                    capture_output=True, text=True, timeout=120,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / "run"
            run("--output", output)
            final = torch.load(output / "last.pt", weights_only=True)
            self.assertEqual(final["epoch"], 2)
            self.assertEqual(final["architecture"]["basis_mode"], "pca_mixing")
            self.assertEqual(len(final["optimizer"]["param_groups"]), 2)
            self.assertIn("train_image_mse", final["history"][0])
            resumed = root / "resumed"
            run("--output", resumed, "--resume", output / "epoch_0001.pt")
            restored = torch.load(resumed / "last.pt", weights_only=True)
            for key in final["model"]:
                torch.testing.assert_close(final["model"][key], restored["model"][key], atol=0, rtol=0)
            self.assertEqual(final["history"], restored["history"])
            result = subprocess.run([
                sys.executable, "-m", "test", "--checkpoint", str(output / "best.pt"),
                "--hr-dir", str(root / "val"), "--output", str(root / "evaluation"),
                "--device", "cpu", "--no-lpips"],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                timeout=120, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((root / "evaluation" / "metrics.json").read_text())["summary"]
            self.assertEqual(summary["images"], 1)
            self.assertLess(summary["dc_max_abs"], 1e-5)


if __name__ == "__main__":
    unittest.main()
