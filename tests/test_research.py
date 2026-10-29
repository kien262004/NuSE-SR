import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
import yaml

from nullspaceSR.analysis.predictability import (ContextCNN, Moments, RidgePredictor,
                                                StreamingRidge, sample_indices, sampled_values, sampled_windows)
from nullspaceSR.analysis.research import DEFAULTS, ResearchRunner
from nullspaceSR.utils import save_checkpoint


def fixtures(root):
    for split, count in (("train", 2), ("val", 1)):
        folder = root / split
        folder.mkdir()
        for i in range(count):
            pixels = bytes((v * 13 + i * 17) % 256 for v in range(16 * 20 * 3))
            Image.frombytes("RGB", (20, 16), pixels).save(folder / f"image_{i}.png")


def tiny_config(root, output):
    config = copy.deepcopy(DEFAULTS)
    config["output"] = str(root / output)
    config["data"].update(train_hr=str(root / "train"), val_hr=str(root / "val"),
                          scale=2, methods=["average_pool"], crop_size=16, repeat=1,
                          batch_size=1, workers=0)
    config["analysis"].update(ranks=[1, 3], contexts=[1, 3], train_samples_per_crop=16,
                              val_samples_per_image=16)
    config["pca"]["repeat"] = 1
    config["cnn"].update(contexts=[3], width=4, epochs=2)
    config["runtime"].update(device="cpu", max_train_batches=2)
    config["benchmark"].update(lr_size=[8, 8], warmup=0, iterations=1)
    return config


class ResearchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_sampled_windows_match_conv_at_boundaries(self):
        lr = torch.rand(2, 3, 5, 7)
        indices = torch.tensor([[0, 6, 34, 17], [0, 28, 34, 14]])
        for context in (1, 3, 5):
            for mode in ("rgb", "same_channel"):
                model = RidgePredictor(3, 2, context, mode)
                windows = sampled_windows(lr, context, indices, mode)
                weights = model.conv.weight.reshape(3 if mode == "same_channel" else 1, -1,
                                                     windows.shape[-1]).transpose(1, 2)
                values = windows @ weights
                values += model.conv.bias.reshape(values.shape[0], 1, -1)
                values = (values.permute(1, 0, 2).flatten(1) if mode == "same_channel" else values.squeeze(0))
                torch.testing.assert_close(values, sampled_values(model(lr), indices), atol=1e-6, rtol=1e-5)

    def test_streaming_ridge_recovers_known_linear_map(self):
        for mode in ("rgb", "same_channel"):
            truth = RidgePredictor(3, 2, 3, mode)
            system = StreamingRidge(3, 2, 3, mode, "cpu")
            for _ in range(3):
                lr = torch.rand(2, 3, 8, 9)
                indices = sample_indices(2, 8, 9, 72, torch.Generator().manual_seed(1))
                system.update(lr, truth(lr).detach(), indices)
            fit = system.solve(1e-10)
            lr = torch.rand(1, 3, 9, 11)
            torch.testing.assert_close(fit(lr), truth(lr), atol=1e-5, rtol=1e-5)

    def test_moments_and_zero_variance(self):
        target = torch.tensor([[1., 2.], [3., 2.]])
        predicted = torch.tensor([[2., 2.], [2., 2.]])
        moments = Moments(2)
        moments.update(target[:1], predicted[:1])
        moments.update(target[1:], predicted[1:])
        rows = moments.rows(1, 2, torch.tensor([1., 2.]))
        self.assertEqual(rows[0]["mse"], 1)
        self.assertEqual(rows[0]["r2"], 0)
        self.assertEqual(rows[0]["train_mean_baseline_mse"], 2)
        self.assertIsNone(rows[1]["r2"])

    def test_cnn_receptive_field(self):
        for context in (1, 3, 5, 9):
            model = ContextCNN(3, 2, context, width=2)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(.1)
            lr = torch.ones(1, 3, 21, 21, requires_grad=True)
            model(lr)[0, 0, 10, 10].backward()
            positions = (lr.grad.abs().sum((0, 1)) > 0).nonzero()
            self.assertEqual((positions.max(0).values - positions.min(0).values + 1).tolist(), [context, context])

    def test_cnn_resume_and_protocol_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures(root)
            config = tiny_config(root, "uninterrupted")
            full = ResearchRunner(config)
            full.ensure_basis()
            full.cnn()
            interrupted_config = tiny_config(root, "resumed")
            interrupted = ResearchRunner(interrupted_config)
            interrupted.ensure_basis()
            def stop_after_save(path, state):
                save_checkpoint(path, state)
                if Path(path).name == "last.pt" and state["epoch"] == 1:
                    raise RuntimeError("simulated interruption")
            with patch("nullspaceSR.analysis.research.save_checkpoint", side_effect=stop_after_save):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    interrupted.cnn()
            interrupted.cnn()
            relative = Path("cnn/average_pool/rgb_r3_seed2026/last.pt")
            a = torch.load(root / "uninterrupted" / relative, weights_only=True)
            b = torch.load(root / "resumed" / relative, weights_only=True)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            changed = copy.deepcopy(config)
            changed["analysis"]["ridge_alpha"] *= 2
            with self.assertRaisesRegex(ValueError, "protocol/data/code changed"):
                ResearchRunner(changed)

    def test_server_cli_all_sweep_and_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures(root)
            config = tiny_config(root, "cli")
            config["data"]["methods"] = ["average_pool", "bicubic"]
            path = root / "research.yaml"
            path.write_text(yaml.safe_dump(config))
            def run(stage):
                result = subprocess.run([sys.executable, "-m", "nullspaceSR.analysis.research",
                                         "--config", str(path), "--smoke", "--stage", stage],
                                        text=True, capture_output=True, timeout=120,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run("all")
            report = root / "cli/reports"
            for name in ("oracle_vs_rank.csv", "coefficient_energy.csv", "r2_vs_context.csv", "quality_vs_rank.csv", "cost.csv"):
                self.assertTrue((report / name).is_file(), name)
            with (report / "quality_vs_rank.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["method"] for row in rows}, {"average_pool", "bicubic"})
            self.assertLess(max(float(r["decomposition_abs"]) for r in rows), 1e-6)
            mismatch = [float(r["measurement_mismatch_mse"]) for r in rows if r["method"] == "bicubic"]
            self.assertGreater(min(mismatch), 1e-6)
            checkpoint = next((root / "cli/cnn").glob("**/last.pt"))
            modified = checkpoint.stat().st_mtime_ns
            run("all")
            self.assertEqual(checkpoint.stat().st_mtime_ns, modified)
            run("sweep")
            run("report")
            with (report / "backbone_sweep.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == "__main__":
    unittest.main()
