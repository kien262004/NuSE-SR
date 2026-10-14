import unittest

import torch
from torch.utils.data import Dataset
from nullspaceSR.data import (make_sample, collate_sr, make_loader, MappingDatasetAdapter,
                             dataset_manifest, set_dataset_epoch, supervised_batch)
from nullspaceSR import AveragePoolSR, BlockNullSpacePCA, NuSESR, IMDN
from nullspaceSR.engine import train_one_epoch
from nullspaceSR.evaluation import evaluate
from custom_data_fixture import ToySRDataset


class DataContractTests(unittest.TestCase):
    def test_independent_metadata_and_unlabelled_inputs(self):
        a = make_sample(torch.zeros(1, 2, 3), sample_id="a", meta={"path": "x"},
                        extras={"mask": torch.ones(2, 3)})
        b = make_sample(torch.ones(1, 2, 3), sample_id="b", meta={"camera": 7})
        batch = collate_sr([a, b])
        self.assertIsNone(batch["target"])
        self.assertEqual(batch["id"], ["a", "b"])
        self.assertEqual(batch["meta"], [{"path": "x"}, {"camera": 7}])
        self.assertIn("mask", batch["extras"][0])
        with self.assertRaises(ValueError):
            supervised_batch(batch, "cpu")

    def test_collate_rejects_ragged_or_mixed_labels(self):
        a = make_sample(torch.zeros(1, 2, 3), sample_id="a")
        b = make_sample(torch.zeros(1, 3, 3), sample_id="b")
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            collate_sr([a, b])
        c = make_sample(torch.zeros(1, 2, 3), torch.zeros(1, 4, 6), sample_id="c")
        with self.assertRaisesRegex(ValueError, "labelled"):
            collate_sr([a, c])

    def test_adapter_and_measurement_validation(self):
        class Legacy(Dataset):
            def __len__(self):
                return 1
            def __getitem__(self, index):
                return {"lr": torch.zeros(1, 2, 3), "hr": torch.ones(1, 4, 6), "name": "legacy"}
        dataset = MappingDatasetAdapter(Legacy())
        batch = next(iter(make_loader(dataset, batch_size=1)))
        self.assertEqual(batch["id"], ["legacy"])
        with self.assertRaisesRegex(ValueError, "A\\(target\\)"):
            supervised_batch(batch, "cpu", channels=1, scale=2, operator=AveragePoolSR(2))

    def test_custom_dataset_runs_engine_without_special_attributes(self):
        dataset = ToySRDataset()
        set_dataset_epoch(dataset, 3)  # Optional hook, must not fail.
        self.assertEqual(dataset_manifest(dataset)["length"], 2)
        loader = make_loader(dataset, batch_size=2)
        decoder = BlockNullSpacePCA.fit([next(iter(loader))["target"]], scale=2, rank=2)
        model = NuSESR(IMDN(1, 2, width=4), decoder)
        optimizer = torch.optim.Adam(model.predictor.parameters())
        metrics = train_one_epoch(model, loader, optimizer, "cpu", log_every=0)
        self.assertGreaterEqual(metrics["train_coefficient_image_mse"], 0)
        summary, _ = evaluate(model, make_loader(dataset, batch_size=1), "cpu")
        self.assertLess(summary["dc_max_abs"], 1e-5)


if __name__ == "__main__":
    unittest.main()
