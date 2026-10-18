"""Tiny image fixtures exercise fitting, training, resume and test CLIs."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from PIL import Image


class PipelineTest(unittest.TestCase):
    def test_nullspace_bicubic_preset_fit_train_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                     "configs/train_div2k_nullspace_bicubic.yaml").read_text())
            self.assertIs(config["model"]["null_space"], True)
            for split in ("train", "val"):
                (root / split).mkdir()
                pixels = bytes((i * 13) % 256 for i in range(16 * 20 * 3))
                Image.frombytes("RGB", (20, 16), pixels).save(root / split / "image.png")
                config["data"][f"{split}_hr"] = (root / split).as_posix()
            basis = root / "basis.pt"
            config["model"].update(scale=2, rank=2, width=4, basis=basis.as_posix())
            config["data"].update(repeat=1, workers=0)
            config["data"]["resize"]["size"] = [8, 8]
            config["train"].update(output=(root / "run").as_posix(), epochs=1, batch_size=1)
            config["runtime"]["device"] = "cpu"
            config["logging"]["enabled"] = False
            config["evaluation"]["lpips"] = False
            path = root / "nullspace_bicubic.yaml"
            path.write_text(yaml.safe_dump(config))
            def run(module, *arguments):
                result = subprocess.run([sys.executable, "-m", module, "--config", str(path),
                                         *map(str, arguments)], capture_output=True, text=True, timeout=90,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run("nullspaceSR.fit_pca", "--output", basis, "--repeat", 1)
            run("train")
            state = torch.load(root / "run" / "last.pt", weights_only=True)
            self.assertTrue(state["architecture"]["null_space"])
            self.assertIn("train_coefficient_image_mse", state["history"][0])
            run("test", "--checkpoint", root / "run" / "last.pt", "--hr-dir", root / "val",
                "--output", root / "results")
            summary = json.loads((root / "results" / "metrics.json").read_text())["summary"]
            self.assertLess(summary["dc_max_abs"], 1e-5)
            self.assertLess(summary["decomposition_abs"], 1e-6)
            self.assertGreater(summary["measurement_mismatch_mse"], 1e-6)
            self.assertAlmostEqual(summary["truncation_mse"], summary["pca_truncation_mse"] +
                                   summary["measurement_mismatch_mse"], places=6)

    def test_repository_bicubic_preset_disables_measurement_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                     "configs/train_div2k_bicubic.yaml").read_text())
            self.assertIs(config["model"]["null_space"], False)
            self.assertEqual(config["data"]["resize"]["method"], "bicubic")
            for split in ("train", "val"):
                (root / split).mkdir()
                pixels = bytes((i * 13) % 256 for i in range(16 * 20 * 3))
                Image.frombytes("RGB", (20, 16), pixels).save(root / split / "image.png")
                config["data"][f"{split}_hr"] = (root / split).as_posix()
            config["model"].update(scale=2, width=4)
            # A leftover basis must never enable PCA when null_space is false.
            config["model"]["basis"] = (root / "missing_basis.pt").as_posix()
            config["data"].update(repeat=1, workers=0)
            config["data"]["resize"]["size"] = [8, 8]
            config["train"].update(output=(root / "run").as_posix(), epochs=1, batch_size=1)
            config["runtime"]["device"] = "cpu"
            config["logging"]["enabled"] = False
            config["evaluation"]["lpips"] = False
            path = root / "bicubic.yaml"
            path.write_text(yaml.safe_dump(config))
            result = subprocess.run([sys.executable, "-m", "train", "--config", str(path)],
                                    capture_output=True, text=True, timeout=90,
                                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = torch.load(root / "run" / "last.pt", weights_only=True)
            self.assertFalse(state["architecture"]["null_space"])
            self.assertIn("train_image_mse", state["history"][0])

    def test_image_space_bicubic_training_resume_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                (root / split).mkdir()
                pixels = bytes((i * 13) % 256 for i in range(16 * 20 * 3))
                Image.frombytes("RGB", (20, 16), pixels).save(root / split / "image.png")
            config = root / "image.yaml"
            # Omitting both null_space and basis must select image mode.
            config.write_text(json.dumps({
                "model": {"scale": 2, "width": 4},
                "data": {"train_hr": (root / "train").as_posix(),
                         "val_hr": (root / "val").as_posix(),
                         "resize": {"method": "bicubic"},
                         "crop_size": 16, "repeat": 1},
                "train": {"epochs": 2, "batch_size": 1, "checkpoint_every": 1},
                "runtime": {"device": "cpu"}}))
            def run(module, *arguments):
                if module in {"train", "test"}:
                    arguments = (*arguments, "--no-lpips")
                result = subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                                        capture_output=True, text=True, timeout=90,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output, resumed = root / "run", root / "resumed"
            run("train", "--config", config, "--output", output)
            state = torch.load(output / "last.pt", weights_only=True)
            self.assertFalse(state["architecture"]["null_space"])
            self.assertIsNone(state["basis_sha256"])
            self.assertIn("train_image_mse", state["history"][0])
            self.assertFalse(any(key.startswith("decoder.") for key in state["model"]))
            run("train", "--config", config, "--output", resumed,
                "--resume", output / "epoch_0001.pt", "--no-null-space",
                "--basis", root / "unused_missing_basis.pt")
            restored = torch.load(resumed / "last.pt", weights_only=True)
            for key in state["model"]:
                torch.testing.assert_close(state["model"][key], restored["model"][key], rtol=0, atol=0)
            run("test", "--config", config, "--checkpoint", output / "best.pt",
                "--hr-dir", root / "val", "--output", root / "results", "--save-images")
            summary = json.loads((root / "results" / "metrics.json").read_text())["summary"]
            self.assertIn("rgb_psnr", summary)
            self.assertNotIn("oracle_psnr", summary)
            self.assertNotIn("dc_max_abs", summary)
            self.assertTrue((root / "results" / "images" / "00001_image.png").is_file())

    def test_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, count in (("train", 2), ("val", 1)):
                folder = root / split
                folder.mkdir()
                for index in range(count):
                    pixels = bytes((i * 13 + index * 17) % 256 for i in range(16 * 20 * 3))
                    Image.frombytes("RGB", (20, 16), pixels).save(folder / f"{index:04d}.png")
            def run(module, *arguments):
                if module in {"train", "test"}:
                    arguments = (*arguments, "--no-lpips")
                result = subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                                        capture_output=True, text=True, timeout=90,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                             "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")])})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            basis, output = root / "basis.pt", root / "run"
            config = root / "pca.yaml"
            config.write_text(f'train_hr: "{(root / "train").as_posix()}"\n'
                              f'output: "{basis.as_posix()}"\n'
                              'scale: 4\nrank: 2\ncrop_size: 16\nrepeat: 1\nbatch_size: 2\n')
            run("nullspaceSR.fit_pca", "--config", config, "--scale", 2)
            arguments = ["--train-hr", root / "train", "--val-hr", root / "val", "--basis", basis,
                         "--output", output, "--scale", 2, "--rank", 2, "--width", 4,
                         "--crop-size", 16, "--repeat", 1, "--batch-size", 2,
                         "--epochs", 2, "--device", "cpu", "--checkpoint-every", 1]
            run("train", *arguments)
            self.assertTrue((output / "best.pt").is_file())
            # Resume epoch one from the same two-epoch schedule and compare.
            last = torch.load(output / "last.pt", weights_only=True)
            self.assertEqual(last["epoch"], 2)
            resumed = root / "resumed"
            run("train", *arguments, "--resume", output / "epoch_0001.pt", "--output", resumed)
            resumed_last = torch.load(resumed / "last.pt", weights_only=True)
            for key in last["model"]:
                torch.testing.assert_close(last["model"][key], resumed_last["model"][key], rtol=0, atol=0)
            # Completed schedule is a supported no-op resume.
            run("train", *arguments, "--resume", output / "last.pt")
            results = root / "evaluation"
            run("test", "--checkpoint", output / "last.pt", "--hr-dir", root / "val",
                "--output", results, "--device", "cpu", "--save-images")
            report = json.loads((results / "metrics.json").read_text())
            self.assertLess(report["summary"]["dc_max_abs"], 1e-5)
            self.assertLess(report["summary"]["decomposition_abs"], 1e-6)
            self.assertTrue((results / "per_image.csv").is_file())
            self.assertTrue((results / "images" / "00001_0000.png").is_file())

    def test_custom_dataset_cli_without_div2k_attributes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def run(module, *arguments):
                if module in {"train", "test"}:
                    arguments = (*arguments, "--no-lpips")
                result = subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                                        capture_output=True, text=True, timeout=90,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                             "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")])})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            dataset = "custom_data_fixture:ToySRDataset"
            basis, output = root / "basis.pt", root / "run"
            run("nullspaceSR.fit_pca", "--train-dataset", dataset, "--output", basis,
                "--scale", 2, "--rank", 2, "--batch-size", 2)
            run("train", "--train-dataset", dataset, "--val-dataset", dataset,
                "--val-dataset-options", '{"split":"val"}', "--basis", basis,
                "--output", output, "--scale", 2, "--rank", 2, "--width", 4,
                "--epochs", 1, "--batch-size", 2, "--device", "cpu")
            run("test", "--test-dataset", dataset, "--test-dataset-options", '{"split":"test"}',
                "--checkpoint", output / "last.pt", "--output", root / "results", "--device", "cpu")
            report = json.loads((root / "results" / "metrics.json").read_text())
            self.assertEqual(report["architecture"]["channels"], 1)
            self.assertLess(report["summary"]["dc_max_abs"], 1e-5)


if __name__ == "__main__":
    unittest.main()
