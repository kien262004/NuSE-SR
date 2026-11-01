import csv
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
import yaml

from nullspaceSR.ablation.run import DEFAULT, Runner, tiled_predict
from nullspaceSR.models import NuSESR
from nullspaceSR.models.ablation import AblationBackbone
from nullspaceSR.pca import BlockNullSpacePCA
from nullspaceSR.utils import model_from_checkpoint, save_checkpoint


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)

    def test_variants_shapes_gradients_and_consistency(self):
        hr = torch.rand(1, 3, 16, 20)
        decoder = BlockNullSpacePCA.fit([hr], scale=2, rank=3)
        lr = torch.nn.functional.interpolate(hr, scale_factor=.5, mode="bicubic", align_corners=False, antialias=True)
        for variant in AblationBackbone.VARIANTS:
            predictor = AblationBackbone(3, 3, 2, variant=variant, width=4, corrected=2, gate_hidden=2)
            model = NuSESR(predictor, decoder)
            if variant not in {"cnn", "linear"}:
                torch.testing.assert_close(predictor(lr), predictor.linear(lr))
            for _ in range(2):
                model.zero_grad()
                (model(lr) - hr).square().mean().backward()
                for p in predictor.parameters():
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-.01)
            self.assertEqual(tuple(predictor(lr).shape), (1, 9, 8, 10))
            torch.testing.assert_close(model.operator(model(lr)), lr, atol=1e-6, rtol=1e-6)
            arch = dict(backbone="ablation", null_space=True, channels=3, rank=3, scale=2, width=4,
                        ablation=dict(variant=variant, width=4, corrected=2, gate_hidden=2, linear_groups="channel"))
            restored = model_from_checkpoint(dict(architecture=arch, model=model.state_dict()))
            torch.testing.assert_close(restored(lr), model(lr), rtol=0, atol=0)

    def test_tiled_local_equivalence_and_global_dependency(self):
        hr = torch.rand(1, 3, 24, 28)
        decoder = BlockNullSpacePCA.fit([hr], scale=2, rank=3)
        lr = torch.rand(1, 3, 12, 14)
        for kind in ("local", "context", "global"):
            predictor = AblationBackbone(3, 3, 2, variant=kind, width=4, corrected=2, gate_hidden=2)
            with torch.no_grad():
                for name, parameter in predictor.named_parameters():
                    parameter.fill_(.1)
                if kind == "global":
                    predictor.gamma.fill_(2)
            model = NuSESR(predictor, decoder).eval()
            whole, tiled = model(lr), tiled_predict(model, lr, 5)
            torch.testing.assert_close(model.operator(tiled), lr, atol=1e-6, rtol=1e-6)
            if kind == "global":
                self.assertGreater((whole - tiled).abs().max().item(), 1e-6)
            else:
                torch.testing.assert_close(whole, tiled, atol=1e-5, rtol=1e-5)

    def test_default_wider_control_is_close_in_parameters(self):
        global_model = AblationBackbone(rank=8, variant="global", width=16, gate_hidden=8)
        wider_model = AblationBackbone(rank=8, variant="wider", width=17, gate_hidden=8)
        global_count = sum(p.numel() for p in global_model.parameters())
        wider_count = sum(p.numel() for p in wider_model.parameters())
        self.assertLess(abs(wider_count - global_count) / global_count, .02)

    def test_cli_train_resume_report_and_regular_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                (root / split).mkdir()
                pixels = bytes((v * 13) % 256 for v in range(16 * 20 * 3))
                Image.frombytes("RGB", (20, 16), pixels).save(root / split / "image.png")
            basis = root / "basis.pt"
            BlockNullSpacePCA.fit([torch.rand(2, 3, 16, 20)], scale=2, rank=3).save(basis)
            cfg = {"output": (root / "runs").as_posix(), "basis": basis.as_posix(),
                   "data": {"train_hr": (root / "train").as_posix(), "val_hr": (root / "val").as_posix(),
                            "scale": 2, "methods": ["average_pool", "bicubic"]}}
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(cfg))
            def run(module, *args):
                result = subprocess.run([sys.executable, "-m", module, *map(str, args)],
                                        capture_output=True, text=True, timeout=120,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            args = ["--config", path, "--smoke", "--device", "cpu"]
            run("nullspaceSR.ablation.run", *args)
            folder = root / "runs" / "runs" / "bicubic" / "g1_global_k3_seed2026"
            self.assertTrue((folder / "DONE").is_file())
            summary = json.loads((folder / "quality.json").read_text())
            self.assertLess(summary["dc_max_abs"], 1e-5)
            self.assertGreater(summary["measurement_mismatch_mse"], 1e-6)
            with (folder / "tiles.csv").open() as stream:
                self.assertLess(float(next(csv.DictReader(stream))["tiled_dc_max_abs"]), 1e-5)
            checkpoint = folder / "last.pt"
            before = checkpoint.stat().st_mtime_ns
            run("nullspaceSR.ablation.run", *args)
            self.assertEqual(before, checkpoint.stat().st_mtime_ns)
            run("nullspaceSR.ablation.run", *args, "--stage", "report")
            with (root / "runs" / "reports" / "runs.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            with (root / "runs" / "reports" / "paired_vs_local.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            run("test", "--checkpoint", folder / "best.pt", "--hr-dir", root / "val",
                "--test-dataset-options", '{"resize":{"method":"bicubic"}}',
                "--output", root / "test_results", "--device", "cpu", "--no-lpips")
            self.assertTrue((root / "test_results" / "metrics.json").is_file())

    def test_interrupted_epoch_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                (root / split).mkdir()
                Image.frombytes("RGB", (20, 16), bytes((v * 13) % 256 for v in range(16 * 20 * 3))).save(
                    root / split / "image.png")
            basis = root / "basis.pt"
            BlockNullSpacePCA.fit([torch.rand(2, 3, 16, 20)], scale=2, rank=3).save(basis)
            cfg = copy.deepcopy(DEFAULT)
            cfg["basis"] = str(basis)
            cfg["data"].update(train_hr=str(root / "train"), val_hr=str(root / "val"), scale=2,
                               methods=["average_pool"], crop_size=16, repeat=1, batch_size=1, workers=0)
            cfg["model"].update(width=4, corrected=2, gate_hidden=2)
            cfg["experiment"].update(ranks=[3], seeds=[2026], epochs=2, val_every=1,
                                     variants=[{"name": "g1_global", "variant": "global"}])
            cfg["runtime"]["device"] = "cpu"
            cfg["evaluation"].update(tile_check_images=0, samples_per_image=16)
            cfg["benchmark"].update(lr_size=[8, 8], warmup=0, iterations=1)
            cfg["output"] = str(root / "full")
            Runner(cfg).run()
            cfg["output"] = str(root / "resumed")
            runner = Runner(cfg)
            def interrupt(path, state):
                save_checkpoint(path, state)
                if Path(path).name == "last.pt" and state["epoch"] == 1:
                    raise RuntimeError("simulated interruption")
            with patch("nullspaceSR.ablation.run.save_checkpoint", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    runner.run()
            runner.run()
            relative = Path("runs/average_pool/g1_global_k3_seed2026/last.pt")
            a = torch.load(root / "full" / relative, weights_only=True)
            b = torch.load(root / "resumed" / relative, weights_only=True)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
