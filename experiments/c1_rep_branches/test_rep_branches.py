"""Kiểm c1 không cần dữ liệu: gộp nhánh cho output giống hệt, và model đã gộp chính là e80.

Chạy: python -m unittest experiments/c1_rep_branches/test_rep_branches.py   (từ gốc repo)
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

import torch  # noqa: E402

from experiments.testing import e80_arch, fake_stats, load_experiment, randomize  # noqa: E402
from nuse_e.models.nuse_e import NuSEE  # noqa: E402
from nuse_e.models.registry import make_nuse  # noqa: E402
from nuse_e.operators import degrade, make_operator  # noqa: E402
from rep_branches import ACBConv, ECBConv  # noqa: E402

IMPL = load_experiment(HERE).IMPL


def small_arch(block):
    return e80_arch((2, 1, 1), 8, block, IMPL if block in ("acb", "ecb") else None)


def small_model(stats, block):
    model = make_nuse(stats, small_arch(block)).double().eval()
    randomize([model])
    return model


class BranchFuseTest(unittest.TestCase):
    def test_fused_conv_matches_three_branches_including_borders(self):
        x = torch.randn(2, 6, 11, 13, dtype=torch.float64)      # cạnh lẻ, không vuông: kiểm cả viền
        for conv in (ACBConv(6), ECBConv(6), ECBConv(6, expand=3)):
            conv = conv.double()
            randomize([conv])
            with torch.no_grad():
                err = (conv(x) - conv.fuse()(x)).abs().max().item()
            self.assertLess(err, 1e-10, type(conv).__name__)


class ModelTest(unittest.TestCase):
    def test_fused_model_is_e80_for_every_budget(self):
        stats = fake_stats()
        y = degrade(torch.rand(1, 3, 40, 56, dtype=torch.float64), "average_pool")
        for block in ("acb", "ecb"):
            model = small_model(stats, block)
            with torch.no_grad():
                before = [model(y, k) for k in model.budgets]
                model.fuse()
                after = [model(y, k) for k in model.budgets]
                plain = NuSEE(stats, small_arch("plain")).double().eval()
                plain.load_state_dict(model.state_dict())        # strict: cùng tên, cùng kích thước với e80
                as_e80 = [plain(y, k) for k in plain.budgets]
            for k, a, b, c in zip(model.budgets, before, after, as_e80):
                self.assertLess((a - b).abs().max().item(), 1e-9, f"{block} K={k}")
                self.assertTrue(torch.equal(b, c), f"{block} K={k}")
            self.assertEqual(sum(p.numel() for p in model.parameters()), model.params_for(model.budgets[-1]))
            self.assertEqual([model.params_for(k) for k in model.budgets],
                             [plain.params_for(k) for k in plain.budgets])

    def test_data_consistency_and_nesting(self):
        stats, op = fake_stats(), make_operator("average_pool")
        y = degrade(torch.rand(1, 3, 32, 48, dtype=torch.float64), "average_pool")
        model = small_model(stats, "ecb")
        with torch.no_grad():
            for k in model.budgets:                               # như T5: basis lưu fp32 nên so tương đối với max|z|
                z = model.detail(model.coefficients(y, k), k)
                err = (op.forward(model(y, k)) - y).abs().max() / z.abs().max().clamp_min(1)
                self.assertLess(err.item(), 1e-6)
            before = model(y, 5).clone()
            for module in [*model.stages[1:], *model.heads[1:]]:
                for p in module.parameters():
                    p.add_(1.0)
            self.assertTrue(torch.equal(before, model(y, 5)))     # exit K=5 không đọc stage / head sau

    def test_training_step_runs(self):
        model = make_nuse(fake_stats(), small_arch("ecb"))
        hr = torch.rand(2, 3, 32, 32)
        loss = model.loss(degrade(hr, "average_pool"), hr)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None for p in model.stages.parameters()))


if __name__ == "__main__":
    unittest.main()
