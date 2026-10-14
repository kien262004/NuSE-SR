"""Tiny image fixtures exercise fitting, training, resume and test CLIs."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


class PipelineTest(unittest.TestCase):
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
                result = subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                                        capture_output=True, text=True, timeout=90,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                             "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")])})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            basis, output = root / "basis.pt", root / "run"
            config = root / "pca.yaml"
            config.write_text(f'train_hr: "{root / "train"}"\noutput: "{basis}"\n'
                              'scale: 4\nrank: 2\ncrop_size: 16\nrepeat: 1\nbatch_size: 2\n')
            run("nullspaceSR.fit_pca", "--config", config, "--scale", 2)
            arguments = ["--train-hr", root / "train", "--val-hr", root / "val", "--basis", basis,
                         "--output", output, "--scale", 2, "--rank", 2, "--width", 4,
                         "--crop-size", 16, "--repeat", 1, "--batch-size", 2,
                         "--epochs", 2, "--device", "cpu", "--checkpoint-every", 1]
            run("nullspaceSR.train", *arguments)
            self.assertTrue((output / "best.pt").is_file())
            # Resume epoch one from the same two-epoch schedule and compare.
            last = torch.load(output / "last.pt", weights_only=True)
            self.assertEqual(last["epoch"], 2)
            resumed = root / "resumed"
            run("nullspaceSR.train", *arguments, "--resume", output / "epoch_0001.pt", "--output", resumed)
            resumed_last = torch.load(resumed / "last.pt", weights_only=True)
            for key in last["model"]:
                torch.testing.assert_close(last["model"][key], resumed_last["model"][key], rtol=0, atol=0)
            # Completed schedule is a supported no-op resume.
            run("nullspaceSR.train", *arguments, "--resume", output / "last.pt")
            results = root / "evaluation"
            run("nullspaceSR.test", "--checkpoint", output / "last.pt", "--hr-dir", root / "val",
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
                result = subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                                        capture_output=True, text=True, timeout=90,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                             "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")])})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            dataset = "custom_data_fixture:ToySRDataset"
            basis, output = root / "basis.pt", root / "run"
            run("nullspaceSR.fit_pca", "--train-dataset", dataset, "--output", basis,
                "--scale", 2, "--rank", 2, "--batch-size", 2)
            run("nullspaceSR.train", "--train-dataset", dataset, "--val-dataset", dataset,
                "--val-dataset-options", '{"split":"val"}', "--basis", basis,
                "--output", output, "--scale", 2, "--rank", 2, "--width", 4,
                "--epochs", 1, "--batch-size", 2, "--device", "cpu")
            run("nullspaceSR.test", "--test-dataset", dataset, "--test-dataset-options", '{"split":"test"}',
                "--checkpoint", output / "last.pt", "--output", root / "results", "--device", "cpu")
            report = json.loads((root / "results" / "metrics.json").read_text())
            self.assertEqual(report["architecture"]["channels"], 1)
            self.assertLess(report["summary"]["dc_max_abs"], 1e-5)


if __name__ == "__main__":
    unittest.main()
